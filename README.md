# 🦀 0xClaw Budget Router Training Module 🚀

This repository contains the open training pipeline for BudgetRouter: dataset construction, project-level train/validation/test splitting, and LoRA SFT training.

The trained LoRA adapter is hosted separately on Hugging Face:

- Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Trained BudgetRouter LoRA: [0xclaw-ai/budgetrouter-qwen3.5-2b-lora](https://huggingface.co/0xclaw-ai/budgetrouter-qwen3.5-2b-lora)
- Redacted dataset: [0xclaw-ai/budgetrouter-hackrep](https://huggingface.co/datasets/0xclaw-ai/budgetrouter-hackrep)

## Setup

```powershell
git clone https://github.com/0xclaw-ai/budget-router-trainer.git
cd budget-router-trainer
conda env create -f environment.yml
conda activate hackrep-budget
pip install -r requirements-train.txt
```

## Download the redacted dataset

The separate [dataset repository](https://huggingface.co/datasets/0xclaw-ai/budgetrouter-hackrep)
contains 56,311 projects and 394,177 phase examples, with project-level splits.
Its dataset card documents source attribution, license scope, weak supervision,
redaction, schemas, and the distinction between the full corpus and the 50k
checkpoint's sampled training inputs.

```powershell
hf download 0xclaw-ai/budgetrouter-hackrep --repo-type dataset --local-dir ./data/budgetrouter-hackrep --include phase_budget_sft_train.jsonl --include phase_budget_sft_val.jsonl --include phase_budget_sft_test.jsonl
```

Set `data.train_file` and `data.val_file` in your local training YAML to
`./data/budgetrouter-hackrep/phase_budget_sft_train.jsonl` and
`./data/budgetrouter-hackrep/phase_budget_sft_val.jsonl`. For the recorded 50k
sampling configuration, use `max_train_samples: 50000`, `max_val_samples: 4096`,
and training seed `20260515`. The existing adapter predates redaction; publishing
the dataset does not retrain it or promise bitwise checkpoint reproduction.

HackRep's official dataset metadata specifies CC BY 4.0. Preserve source
attribution and change notices, and consult the dataset's `NOTICE.md` for the
scope of that license and source-project rights. The code license below does
not relicense third-party README excerpts.

## Build Dataset from the upstream source

Download `HackRep.parquet` from the [official HackRep dataset record](https://zenodo.org/records/17572684) into `./data/`, then run:

```powershell
python scripts/prepare_hackrep_budget_data.py --config config/data.example.yaml --parquet-path ./data/HackRep.parquet --no-extract
```

This writes project metadata and SFT JSONL splits under `./outputs/`.

Alternatively, place a full-record archive named `17572684.zip` in `./data/`
and omit `--parquet-path` and `--no-extract` to use the existing extraction path.

## Train LoRA

Download [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) locally and update `config/train_qwen35_2b_lora.example.yaml` so `model.model_path` points to that directory.

```powershell
python training/train_budget_sft.py --config config/train_qwen35_2b_lora.example.yaml
```

The adapter is written under `./checkpoints/`. Model weights and generated datasets are intentionally ignored by Git.

## Export a redacted dataset copy

Before sharing generated data, run the separate sanitization utility:

```bash
python scripts/sanitize_dataset.py ./outputs ./sanitized-data --backup-dir ./private-backups/original-outputs
```

The utility verifies a byte-for-byte backup before exporting JSON/JSONL files to
an empty destination. It redacts recognized API-key formats, secret assignments,
bearer tokens, private-key blocks, URL credentials, and local machine paths.
The original files remain unchanged. The report records counts and SHA-256
checksums without recording any credential values.

Sample order, project splits, labels, numeric features, and assistant control
tokens are preserved. A suspected secret inside a protected label or metadata
field stops the export for review. Old byte-offset index files are omitted;
the existing training loader rebuilds them for the redacted files.

To train on the copy, set `data.train_file` and `data.val_file` in your local
training config to `./sanitized-data/phase_budget_sft_train.jsonl` and
`./sanitized-data/phase_budget_sft_val.jsonl`. The training implementation and
existing model weights are unchanged. Pattern-based redaction does not establish
whether a credential was live or detect every possible secret format.

## Files

- `scripts/prepare_hackrep_budget_data.py`: filters HackRep projects, creates weak-supervision phase labels, and writes train/validation/test JSONL splits.
- `training/train_budget_sft.py`: Qwen3.5-2B LoRA SFT training script.
- `scripts/sanitize_dataset.py`: backup verification and separate data sanitization.
- `config/*.example.yaml`: safe relative-path example configs.

## Export a portable adapter

Training checkpoints record the local base-model path. Export a separate release
copy with a portable Hub model ID before uploading it:

```bash
python scripts/export_adapter.py ./checkpoints/qwen35_2b_budget_lora/final ./model-release
```

The export copies `adapter_model.safetensors` unchanged and sets
`base_model_name_or_path` to `Qwen/Qwen3.5-2B`. The runtime uses the base model's
tokenizer. Add a model card to the export directory before publication.

Generated datasets, manifests, resolved training configs, checkpoints, and logs
are ignored by Git. Unredacted exports may contain source repository text or
machine-specific paths. This repository distributes preparation code and
relative-path example configs; the reviewed redacted data is hosted in the
separate [dataset repository](https://huggingface.co/datasets/0xclaw-ai/budgetrouter-hackrep).

The training and dataset construction algorithms are unchanged. Installed wheels
also include the scripts and example configs under `budget_router_trainer`;
for example, `python -m budget_router_trainer.scripts.prepare_hackrep_budget_data --help`.

## Related resources

- [0xClaw](https://github.com/0xclaw-ai/0xClaw)
- [Inference component](https://github.com/0xclaw-ai/budget-router)
- [DeadlineFramer](https://github.com/0xclaw-ai/deadline-framer)
- [Paper / preprint](https://spiral.imperial.ac.uk/entities/publication/7a460786-fba9-4b42-b9a0-d128429f67e4)

## License

MIT for this repository's code. The HackRep dataset collection is licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); source-project
material and models may have separate rights and terms. See the dataset card
and its attribution notice before reuse.
