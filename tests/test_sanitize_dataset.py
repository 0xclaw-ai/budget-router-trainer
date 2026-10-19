import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("sanitize_dataset", Path(__file__).parents[1] / "scripts/sanitize_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("value", [
    "sk-" + "Ab9" * 12,
    "hf_" + "Ab9" * 12,
    "github_pat_" + "Ab9" * 15,
    "AIza" + "Ab9" * 12,
    "OPENAI_API_KEY=" + "Ab9" * 12,
    "_API_KEY=" + "Ab9" * 12,
    "serviceApiKey=" + "Ab9" * 12,
    "Authorization: Bearer " + "Ab9" * 12,
    "https://demo:" + "Ab9" * 12 + "@example.org",
    '"' + "C:" + "\\Users\\Sample Person\\project\\config.json" + '"',
    "/" + "Users/sample/project/config.json",
])
def test_secrets_and_local_paths_are_redacted_idempotently(value):
    clean, counts = module.sanitize_text(value)
    assert clean != value
    assert sum(counts.values()) > 0
    assert "REDACTED" in clean
    assert module.sanitize_text(clean) == (clean, Counter())


@pytest.mark.parametrize("closed", [True, False])
def test_private_key_body_is_removed_even_if_excerpt_is_truncated(closed):
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    footer = "-----END " + "RSA PRIVATE KEY-----"
    value = header + "\n" + "AbCd0123" * 12 + ("\n" + footer if closed else "")
    clean, counts = module.sanitize_text(value)
    assert clean == "[REDACTED_PRIVATE_KEY]"
    assert counts == {"private_key": 1}


def test_labels_metadata_and_numeric_features_are_preserved():
    original = {"messages": [
        {"role": "user", "content": "key: sk-" + "Ab9" * 12},
        {"role": "assistant", "content": "<PHASE_CODING><TOKEN_1.0X><TIME_HIGH><THINK_LOW><REPAIR_1>"},
    ], "metadata": {"project_id": "example/project", "split": "train", "phase_score": 1.2}}
    clean = module.sanitize_record(original, Counter())
    assert clean["messages"][1] == original["messages"][1]
    assert clean["metadata"] == original["metadata"]
    assert clean["messages"][0] != original["messages"][0]


def test_short_sk_repository_name_is_not_a_provider_key():
    value = "example/sk-v1-sample-project"
    assert module.sanitize_text(value) == (value, Counter())


def test_protected_fields_fail_without_printing_secret():
    value = "sk-" + "Ab9" * 12
    with pytest.raises(ValueError) as error:
        module.sanitize_record({"metadata": {"private_value": value}}, Counter())
    assert value not in str(error.value)


def test_backup_export_and_stale_index_handling(tmp_path):
    source, destination, backup = [tmp_path / part for part in ("source", "output", "backup")]
    source.mkdir()
    row = {"input": "API_KEY=" + "Ab9" * 12, "target": "<PHASE_CODING>", "split": "train"}
    raw = (json.dumps(row) + "\n").encode()
    (source / "train.jsonl").write_bytes(raw)
    (source / "train.jsonl.idx").write_text("0\n")
    report = module.sanitize_dataset(source, destination, backup)
    assert (source / "train.jsonl").read_bytes() == raw
    assert (backup / "train.jsonl").read_bytes() == raw
    assert not (destination / "train.jsonl.idx").exists()
    assert json.loads((destination / "train.jsonl").read_text())["target"] == row["target"]
    assert report["files"]["train.jsonl"]["changed_rows"] == 1
    with pytest.raises(ValueError, match="empty"):
        module.sanitize_dataset(source, destination, backup)


def test_mismatched_backup_is_never_overwritten(tmp_path):
    source, destination, backup = [tmp_path / part for part in ("source", "output", "backup")]
    source.mkdir()
    backup.mkdir()
    (source / "train.jsonl").write_text('{}\n')
    (backup / "train.jsonl").write_text('original recovery copy\n')
    with pytest.raises(ValueError, match="Backup does not match"):
        module.sanitize_dataset(source, destination, backup)
    assert (backup / "train.jsonl").read_text() == 'original recovery copy\n'
    assert not destination.exists()
