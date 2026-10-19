"""Export portable adapter metadata and unchanged weights for publication."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def export_adapter(source: Path, destination: Path, base_model: str) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate destination to preserve the training checkpoint")
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    weights = source / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if ":" in base_model or "\\" in base_model or len(base_model.split("/")) != 2:
        raise ValueError("base_model must be a Hub model ID such as Qwen/Qwen3.5-2B")
    config["base_model_name_or_path"] = base_model
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, destination / weights.name)
    (destination / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    args = parser.parse_args()
    export_adapter(args.source, args.destination, args.base_model)


if __name__ == "__main__":
    main()
