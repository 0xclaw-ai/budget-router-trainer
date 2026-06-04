from __future__ import annotations

import os

# Windows conda environments can load both libomp.dll and libiomp5md.dll
# through different binary packages. Set this before importing torch/numpy-backed
# libraries so a quick local SFT run can proceed. For final large runs, a clean
# CUDA/PyTorch environment with a single OpenMP runtime is still preferable.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class JsonlOffsetDataset(Dataset):
    """Random-access JSONL dataset backed by byte offsets.

    This avoids loading the 900MB+ SFT train file into memory.
    """

    def __init__(self, path: str | Path, max_samples: int | None = None, sample_seed: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.offsets = self._build_offsets()
        if max_samples is not None and len(self.offsets) > max_samples:
            rng = random.Random(sample_seed)
            self.offsets = rng.sample(self.offsets, max_samples)

    def _build_offsets(self) -> list[int]:
        idx_path = self.path.with_suffix(self.path.suffix + ".idx")
        meta_path = self.path.with_suffix(self.path.suffix + ".idx.meta.json")
        current_meta = {
            "path": str(self.path),
            "size": self.path.stat().st_size,
            "mtime": self.path.stat().st_mtime,
        }
        if idx_path.exists() and meta_path.exists():
            try:
                saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if saved_meta == current_meta:
                    return [int(x) for x in idx_path.read_text(encoding="utf-8").splitlines()]
            except Exception:
                pass

        offsets: list[int] = []
        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)

        idx_path.write_text("\n".join(str(x) for x in offsets), encoding="utf-8")
        meta_path.write_text(json.dumps(current_meta), encoding="utf-8")
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with self.path.open("rb") as f:
            f.seek(self.offsets[idx])
            return json.loads(f.readline().decode("utf-8"))


@dataclass
class SFTCollator:
    tokenizer: Any
    max_seq_len: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        attention_list: list[torch.Tensor] = []

        for item in batch:
            messages = item["messages"]
            prompt_messages = messages[:-1]

            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

            if self.tokenizer.eos_token_id is not None:
                full_ids = full_ids + [self.tokenizer.eos_token_id]

            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[-self.max_seq_len :]
                prompt_len = min(len(prompt_ids), self.max_seq_len)
                # If the prompt was truncated away, only supervise the final assistant tokens.
                prompt_len = min(prompt_len, max(0, len(full_ids) - 16))
            else:
                prompt_len = min(len(prompt_ids), len(full_ids))

            ids = torch.tensor(full_ids, dtype=torch.long)
            labels = ids.clone()
            labels[:prompt_len] = -100
            attention = torch.ones_like(ids)

            input_ids_list.append(ids)
            labels_list.append(labels)
            attention_list.append(attention)

        pad_id = self.tokenizer.pad_token_id
        max_len = max(x.size(0) for x in input_ids_list)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, (ids, labs, attn) in enumerate(zip(input_ids_list, labels_list, attention_list)):
            input_ids[i, : ids.size(0)] = ids
            labels[i, : labs.size(0)] = labs
            attention_mask[i, : attn.size(0)] = attn

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, output_dir: Path, step: int) -> None:
    ckpt = output_dir / f"step-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)


def cuda_memory_report(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    free, total = torch.cuda.mem_get_info(device)
    return (
        f" cuda_alloc={allocated:.2f}G"
        f" cuda_reserved={reserved:.2f}G"
        f" cuda_max_reserved={max_reserved:.2f}G"
        f" cuda_free={free / 1024**3:.2f}/{total / 1024**3:.2f}G"
    )


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
            outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.output_dir is not None:
        cfg["output"]["output_dir"] = str(args.output_dir)
    if args.log_dir is not None:
        cfg["output"]["log_dir"] = str(args.log_dir)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available; training will be very slow.")

    model_path = cfg["model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if device.type == "cuda" and cfg["train"].get("bf16", True) and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda" and cfg["train"].get("fp16", False):
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.config.use_cache = False
    if cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    train_ds = JsonlOffsetDataset(cfg["data"]["train_file"], cfg["data"].get("max_train_samples"), sample_seed=seed)
    val_ds = JsonlOffsetDataset(cfg["data"]["val_file"], cfg["data"].get("max_val_samples"), sample_seed=seed + 1)
    print(f"Train rows: {len(train_ds):,}; Val rows: {len(val_ds):,}")

    collator = SFTCollator(tokenizer=tokenizer, max_seq_len=int(cfg["data"]["max_seq_len"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    grad_accum = int(cfg["train"]["gradient_accumulation_steps"])
    epochs = int(cfg["train"]["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_steps * float(cfg["train"]["warmup_ratio"]))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(cfg["output"]["output_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    writer = SummaryWriter(log_dir=str(log_dir))

    scaler_enabled = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    global_step = 0
    running_loss = 0.0
    running_items = 0
    start = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for micro_step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            raw_loss = float(loss.detach().cpu()) * grad_accum
            running_loss += raw_loss
            running_items += 1
            pbar.set_postfix(loss=f"{raw_loss:.4f}", step=global_step)

            is_update_step = micro_step % grad_accum == 0 or micro_step == len(train_loader)
            if not is_update_step:
                continue

            if scaler_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["max_grad_norm"]))

            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            avg_loss = running_loss / max(1, running_items)
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train/epoch", epoch + micro_step / max(1, len(train_loader)), global_step)
            running_loss = 0.0
            running_items = 0

            if global_step % int(cfg["train"]["log_every_steps"]) == 0:
                elapsed = max(1e-6, time.time() - start)
                print(
                    f"step={global_step} loss={avg_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"steps/s={global_step / elapsed:.3f}"
                    f"{cuda_memory_report(device)}"
                )

            if global_step % int(cfg["train"]["eval_every_steps"]) == 0:
                val_loss = evaluate(model, val_loader, device, dtype, max_batches=64)
                writer.add_scalar("val/loss", val_loss, global_step)
                writer.add_scalar("val/ppl", math.exp(min(20, val_loss)), global_step)
                print(f"eval step={global_step} val_loss={val_loss:.4f}{cuda_memory_report(device)}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    print(f"after eval empty_cache{cuda_memory_report(device)}")

            if global_step % int(cfg["train"]["save_every_steps"]) == 0:
                save_checkpoint(model, tokenizer, output_dir, global_step)

    final_val = evaluate(model, val_loader, device, dtype, max_batches=None)
    writer.add_scalar("val/final_loss", final_val, global_step)
    writer.close()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final val loss: {final_val:.4f}")
    print(f"Saved final LoRA adapter to: {final_dir}")
    print(f"TensorBoard logs: {log_dir}")


if __name__ == "__main__":
    main()
