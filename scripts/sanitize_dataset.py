"""Export a redacted dataset copy without changing labels or training code."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PRIVATE_KEY = re.compile(
    r"-----BEGIN (?P<kind>(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY)-----"
    r"[\s\S]*?(?:-----END (?P=kind)-----|\Z)"
)
API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{16,}|"
    r"gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|"
    r"gsk_[A-Za-z0-9]{16,}|AIza[A-Za-z0-9_-]{30,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{16,})\b"
)
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
ASSIGNMENT = re.compile(
    r'''(?i)((?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|secret[_-]?key)["']?\s*[=:]\s*["']?)([A-Za-z0-9_./+\-=]{12,})'''
)
BEARER = re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9_./+\-=]{20,})")
URL_AUTH = re.compile(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@")
QUOTED_PATH = re.compile(
    r'''(?P<quote>["'])(?P<path>(?:[A-Za-z]:[\\/]|/(?:Users|home|root)/)[^\r\n"']+)(?P=quote)'''
)
LOCAL_PATH = re.compile(
    r'''(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/(?:Users|home|root)/|/mnt/[a-z]/Users/)[^\s<>"'`|,;\]\)}]+'''
)
PROTECTED = {"metadata", "project_id", "phase", "split", "target", "label",
             "features", "phase_budget_labels"}
TOKEN_MARKERS = ("sk-", "hf_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
                 "github_pat_", "gsk_", "AIza", "AKIA", "ASIA", "xox")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sanitize_text(text: str) -> tuple[str, Counter]:
    counts = Counter()

    def apply(pattern, replacement, kind):
        nonlocal text
        text, count = pattern.subn(replacement, text)
        counts[kind] += count

    if "PRIVATE KEY-----" in text:
        apply(PRIVATE_KEY, "[REDACTED_PRIVATE_KEY]", "private_key")
    if any(marker in text for marker in TOKEN_MARKERS):
        apply(API_KEY, "[REDACTED_API_KEY]", "api_key")
    if "eyJ" in text:
        apply(JWT, "[REDACTED_TOKEN]", "jwt")
    lower = text.lower()
    if any(word in lower for word in ("key", "token", "password", "secret")):
        apply(ASSIGNMENT, lambda m: m.group(1) + "[REDACTED_SECRET]", "secret_assignment")
    if "bearer" in lower:
        apply(BEARER, lambda m: m.group(1) + "[REDACTED_TOKEN]", "bearer")
    if "://" in text:
        apply(URL_AUTH, lambda m: m.group(1) + "[REDACTED_CREDENTIAL]@", "url_credentials")
    if "\\" in text or ":/" in text or any(p in text for p in ("/home/", "/Users/", "/root/", "/mnt/")):
        apply(QUOTED_PATH, lambda m: m.group("quote") + "[REDACTED_LOCAL_PATH]" + m.group("quote"), "local_path")
        apply(LOCAL_PATH, "[REDACTED_LOCAL_PATH]", "local_path")
    return text, +counts


def sanitize_record(value: Any, counts: Counter) -> Any:
    if isinstance(value, str):
        clean, changes = sanitize_text(value)
        counts.update(changes)
        return clean
    if isinstance(value, list):
        return [sanitize_record(item, counts) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PROTECTED or (key == "content" and value.get("role") == "assistant"):
                candidate_counts = Counter()
                clean = sanitize_record(item, candidate_counts)
                if clean != item:
                    raise ValueError(f"Sensitive content found in protected field {key!r}; refusing to change labels")
                result[key] = item
            else:
                result[key] = sanitize_record(item, counts)
        return result
    return value


def sanitize_dataset(source: Path, destination: Path, backup: Path) -> dict:
    source, destination, backup = (p.resolve() for p in (source, destination, backup))
    for left, right in ((source, destination), (source, backup), (destination, backup)):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Source, destination, and backup must be separate, non-nested directories")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty; existing exports are never overwritten")
    files = sorted(p for p in source.iterdir() if p.is_file())
    if not files:
        raise ValueError("Source contains no files")
    if any(p.is_symlink() for p in source.iterdir()):
        raise ValueError("Source symlinks are not supported")
    backup.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for path in files:
        original_hash = sha256(path)
        saved = backup / path.name
        if not saved.exists():
            shutil.copy2(path, saved)
        if sha256(saved) != original_hash:
            raise ValueError(f"Backup does not match source: {path.name}")
        source_hashes[path.name] = original_hash
    print(f"Verified backups for {len(files)} source files", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    report = {"source_sha256": source_hashes, "files": {}, "omitted_sidecars": [],
              "labels_preserved": True, "source_unchanged": True}
    for path in files:
        if path.suffix not in {".json", ".jsonl"} or ".idx" in path.name:
            report["omitted_sidecars"].append(path.name)
            continue
        counts = Counter()
        rows = changed_rows = 0
        target = destination / path.name
        temporary = target.with_name(target.name + ".tmp")
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as stream, temporary.open("w", encoding="utf-8", newline="\n") as output:
                for line in stream:
                    if not line.strip():
                        output.write(line)
                        continue
                    original = json.loads(line)
                    clean = sanitize_record(original, counts)
                    rows += 1
                    changed_rows += clean != original
                    output.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n" if clean != original else line)
                    if rows % 100000 == 0:
                        print(f"{path.name}: {rows} rows processed", flush=True)
        else:
            original = json.loads(path.read_text(encoding="utf-8"))
            clean = sanitize_record(original, counts)
            rows, changed_rows = 1, int(clean != original)
            temporary.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(target)
        if sha256(path) != source_hashes[path.name]:
            raise RuntimeError(f"Source changed during export: {path.name}")
        report["files"][path.name] = {"rows": rows, "changed_rows": changed_rows,
                                      "redactions": dict(counts), "sha256": sha256(target)}
        print(f"{path.name}: {changed_rows}/{rows} rows redacted", flush=True)
    (destination / "sanitization-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    sanitize_dataset(args.source, args.destination, args.backup_dir)


if __name__ == "__main__":
    main()
