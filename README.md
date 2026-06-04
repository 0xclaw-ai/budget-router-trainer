# BudgetRouter Training

This repository contains the open training pipeline for BudgetRouter: dataset construction, project-level train/validation/test splitting, and LoRA SFT training.

The trained LoRA adapter is hosted separately on Hugging Face:

- Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Trained BudgetRouter LoRA: [0xclaw-ai/budgetrouter-qwen3.5-2b-lora](https://huggingface.co/0xclaw-ai/budgetrouter-qwen3.5-2b-lora)

## Setup

```powershell
conda env create -f environment.yml
conda activate hackrep-budget
pip install -r requirements-train.txt
```

## Build Dataset

Place the HackRep archive under `./data/`, then run:

```powershell
python scripts/prepare_hackrep_budget_data.py --config config/data.example.yaml
```

This writes project metadata and SFT JSONL splits under `./outputs/`.

## Train LoRA

Download [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) locally and update `config/train_qwen35_2b_lora.example.yaml` so `model.model_path` points to that directory.

```powershell
python training/train_budget_sft.py --config config/train_qwen35_2b_lora.example.yaml
```

The adapter is written under `./checkpoints/`. Model weights and generated datasets are intentionally ignored by Git.

## Files

- `scripts/prepare_hackrep_budget_data.py`: filters HackRep projects, creates weak-supervision phase labels, and writes train/validation/test JSONL splits.
- `training/train_budget_sft.py`: Qwen3.5-2B LoRA SFT training script.
- `config/*.example.yaml`: safe relative-path example configs.
