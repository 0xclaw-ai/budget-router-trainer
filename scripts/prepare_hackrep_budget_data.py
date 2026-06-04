from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required to read HackRep.parquet. "
        "Create the conda env with `conda env create -f environment.yml` first."
    ) from exc


PHASES = [
    "research",
    "idea",
    "selection",
    "planning",
    "coding",
    "testing",
    "documentation",
]

PHASE_PRIORS = {
    "research": 0.85,
    "idea": 0.80,
    "selection": 0.95,
    "planning": 0.95,
    "coding": 1.15,
    "testing": 1.05,
    "documentation": 0.75,
}

PHASE_TIME_PRIORS = {
    "research": 0.90,
    "idea": 0.75,
    "selection": 0.95,
    "planning": 0.90,
    "coding": 1.15,
    "testing": 1.20,
    "documentation": 0.70,
}

KEYWORDS = {
    "research": [
        "research",
        "paper",
        "dataset",
        "benchmark",
        "experiment",
        "evaluation",
        "model",
        "training",
        "fine-tune",
        "finetune",
        "hypothesis",
        "baseline",
    ],
    "ml": [
        "machine learning",
        "deep learning",
        "neural",
        "llm",
        "transformer",
        "computer vision",
        "nlp",
        "classification",
        "regression",
        "embedding",
        "rag",
        "agent",
        "reinforcement",
    ],
    "product": [
        "user",
        "customer",
        "workflow",
        "dashboard",
        "app",
        "mobile",
        "web",
        "demo",
        "prototype",
        "feature",
        "use case",
        "problem",
        "solution",
    ],
    "architecture": [
        "architecture",
        "backend",
        "frontend",
        "database",
        "server",
        "client",
        "api",
        "service",
        "microservice",
        "pipeline",
        "docker",
        "deploy",
        "cloud",
    ],
    "external_api": [
        "openai",
        "anthropic",
        "deepseek",
        "qwen",
        "google",
        "maps",
        "firebase",
        "supabase",
        "stripe",
        "twilio",
        "slack",
        "discord",
        "spotify",
        "notion",
        "github api",
        "weather",
        "aws",
        "azure",
        "gcp",
    ],
    "testing": [
        "test",
        "pytest",
        "unittest",
        "jest",
        "vitest",
        "ci",
        "github actions",
        "coverage",
        "bug",
        "fix",
        "debug",
        "error",
        "fail",
    ],
    "documentation": [
        "install",
        "installation",
        "usage",
        "setup",
        "run",
        "quickstart",
        "demo",
        "screenshots",
        "license",
        "contributing",
        "documentation",
        "docs",
    ],
}

COMMIT_KEYWORDS = {
    "fix": ["fix", "bug", "debug", "error", "fail", "hotfix", "patch"],
    "test": ["test", "ci", "coverage", "spec"],
    "doc": ["doc", "readme", "docs", "comment"],
    "feature": ["feat", "feature", "implement", "add", "create"],
    "refactor": ["refactor", "cleanup", "clean up", "restructure"],
}

SELECTED_COLUMNS = [
    "html_url",
    "full_name",
    "description",
    "topics",
    "language",
    "size",
    "stargazers_count",
    "forks_count",
    "open_issues_count",
    "homepage",
    "has_pages",
    "license",
    "fork",
    "archived",
    "disabled",
    "private",
    "readme",
    "commit_shas",
    "commit_messages",
    "estimated_duration",
    "contributors",
    "scientific_probability",
    "scientific_classification",
]


def read_yaml(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def default_config() -> dict[str, Any]:
    return {
        "input": {
            "zip_path": "./data/17572684.zip",
            "parquet_name": "HackRep.parquet",
            "work_dir": "./work/hackrep_budget_work",
        },
        "output": {
            "output_dir": "./outputs",
            "project_jsonl": "clean_projects.jsonl",
            "sft_prefix": "phase_budget_sft",
            "flat_prefix": "phase_budget_flat",
            "manifest": "manifest.json",
        },
        "filters": {
            "min_readme_chars": 200,
            "max_readme_chars": 30000,
            "min_commits": 3,
            "min_duration_hours": 1,
            "max_duration_hours": 120,
            "drop_forks": True,
            "drop_archived": True,
            "drop_disabled": True,
            "require_language": True,
            "require_readme": True,
        },
        "prompt": {"readme_excerpt_chars": 1200, "include_commit_keywords": True},
        "splits": {"train_pct": 90, "val_pct": 5, "test_pct": 5, "seed": 20260515},
        "processing": {"batch_size": 1024, "limit": None, "keep_invalid_projects": False},
        "weak_supervision": {
            "enabled": True,
            "min_phase_confidence": 0.0,
            "drop_low_confidence": False,
        },
        "budget_bins": {
            "token": [
                [0.375, "TOKEN_0.25X"],
                [0.625, "TOKEN_0.5X"],
                [0.875, "TOKEN_0.75X"],
                [1.125, "TOKEN_1.0X"],
                [999.0, "TOKEN_1.25X"],
            ],
            "time": [
                [0.65, "TIME_LOW"],
                [0.95, "TIME_MEDIUM"],
                [1.20, "TIME_HIGH"],
                [999.0, "TIME_CRITICAL"],
            ],
        },
    }


def clean_text(value: Any, max_chars: int | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\w.\-+]+@[\w.\-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + " ..."
    return text


def safe_literal(value: Any) -> Any:
    if value is None or isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def as_list(value: Any) -> list[Any]:
    value = safe_literal(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return [value] if value.strip() else []
    return [value]


def as_dict(value: Any) -> dict[str, Any]:
    value = safe_literal(value)
    if isinstance(value, dict):
        return value
    return {}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def count_any(text: str, words: Iterable[str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(word.lower()) + r"\b", lowered)) for word in words)


def log_norm(value: float, scale: float) -> float:
    value = max(0.0, value)
    return min(1.0, math.log1p(value) / math.log1p(scale))


def ratio_to_bin(value: float, bins: list[list[Any]]) -> str:
    for threshold, label in bins:
        if value <= float(threshold):
            return str(label)
    return str(bins[-1][1])


def confidence_bin(value: float) -> str:
    if value < 0.20:
        return "very_low"
    if value < 0.40:
        return "low"
    if value < 0.60:
        return "medium"
    if value < 0.80:
        return "high"
    return "very_high"


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def extract_commit_info(commit_messages: Any) -> dict[str, Any]:
    commits = as_list(commit_messages)
    dates: list[datetime] = []
    messages: list[str] = []

    for item in commits:
        if not isinstance(item, dict):
            continue
        commit = item.get("commit")
        if not isinstance(commit, dict):
            commit = item
        msg = commit.get("message") if isinstance(commit, dict) else None
        if isinstance(msg, str):
            messages.append(msg[:400])

        date = None
        for parent_key in ("committer", "author"):
            parent = commit.get(parent_key) if isinstance(commit, dict) else None
            if isinstance(parent, dict):
                date = date or parent.get("date")
        dt = parse_datetime(date)
        if dt:
            dates.append(dt)

    joined = " ".join(messages).lower()
    keyword_counts = {
        key: count_any(joined, words) for key, words in COMMIT_KEYWORDS.items()
    }
    span_hours = None
    if len(dates) >= 2:
        span = max(dates) - min(dates)
        span_hours = max(0.0, span.total_seconds() / 3600.0)

    return {
        "commit_count": len(commits),
        "commit_message_count": len(messages),
        "commit_span_hours": span_hours,
        "commit_keyword_counts": keyword_counts,
    }


def extract_duration(estimated_duration: Any, commit_span_hours: float | None) -> float | None:
    duration = as_dict(estimated_duration)
    raw = duration.get("duration_hours")
    if raw is not None:
        value = to_float(raw, default=float("nan"))
        if not math.isnan(value):
            return value
    return commit_span_hours


def duration_bin(hours: float | None) -> str:
    if hours is None or math.isnan(hours):
        return "unknown"
    if hours < 8:
        return "short"
    if hours < 24:
        return "medium"
    if hours < 72:
        return "long"
    return "very_long"


def split_for_name(name: str, cfg: dict[str, Any]) -> str:
    split_cfg = cfg["splits"]
    train_pct = int(split_cfg["train_pct"])
    val_pct = int(split_cfg["val_pct"])
    seed = str(split_cfg.get("seed", 0))
    digest = hashlib.sha1((seed + "::" + name).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def project_features(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    readme = clean_text(row.get("readme"))
    description = clean_text(row.get("description"), 1000)
    topics = [clean_text(x, 80) for x in as_list(row.get("topics"))]
    topic_text = " ".join(topics)
    text_all = " ".join([description, topic_text, readme])

    commit_info = extract_commit_info(row.get("commit_messages"))
    commit_count = commit_info["commit_count"]
    duration_hours = extract_duration(
        row.get("estimated_duration"), commit_info.get("commit_span_hours")
    )
    contributors = as_list(row.get("contributors"))
    readme_chars = len(readme)
    readme_words = len(re.findall(r"\w+", readme))

    keyword_counts = {
        key: count_any(text_all, words) for key, words in KEYWORDS.items()
    }

    markdown_headings = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+", readme))
    code_blocks = readme.count("```")
    links = len(re.findall(r"\[[^\]]+\]\([^)]+\)", readme)) + readme.count("[URL]")
    images = readme.lower().count("![")
    bullet_lines = len(re.findall(r"(?m)^\s*[-*+]\s+", readme))

    feature_count = min(30, bullet_lines + markdown_headings)
    has_demo = bool(row.get("homepage")) or to_bool(row.get("has_pages")) or bool(
        re.search(r"\b(demo|live|video|youtube|screenshot|preview|deploy)\b", readme, re.I)
    )
    has_install = bool(re.search(r"\b(install|installation|setup|quickstart)\b", readme, re.I))
    has_usage = bool(re.search(r"\b(usage|run|start|how to use|example)\b", readme, re.I))
    has_tests = bool(
        re.search(r"\b(test|pytest|unittest|jest|vitest|coverage|github actions|ci)\b", readme, re.I)
    )
    has_license = bool(row.get("license")) and str(row.get("license")).lower() != "nan"

    repo_size = to_float(row.get("size"))
    scientific_prob = to_float(row.get("scientific_probability"))

    return {
        "full_name": clean_text(row.get("full_name"), 300),
        "html_url": clean_text(row.get("html_url"), 500),
        "description": description,
        "topics": topics[:30],
        "language": clean_text(row.get("language"), 80),
        "repo_size_kb": repo_size,
        "stars": int(to_float(row.get("stargazers_count"))),
        "forks": int(to_float(row.get("forks_count"))),
        "open_issues": int(to_float(row.get("open_issues_count"))),
        "readme_excerpt": clean_text(
            readme, int(cfg["prompt"]["readme_excerpt_chars"])
        ),
        "readme_chars": readme_chars,
        "readme_words": readme_words,
        "markdown_headings": markdown_headings,
        "code_blocks": code_blocks,
        "links": links,
        "images": images,
        "feature_count": feature_count,
        "has_demo": has_demo,
        "has_install": has_install,
        "has_usage": has_usage,
        "has_tests": has_tests,
        "has_license": has_license,
        "commit_count": commit_count,
        "commit_message_count": commit_info["commit_message_count"],
        "commit_span_hours": commit_info["commit_span_hours"],
        "duration_hours": duration_hours,
        "duration_bin": duration_bin(duration_hours),
        "contributors_count": len(contributors),
        "scientific_probability": scientific_prob,
        "scientific_classification": clean_text(row.get("scientific_classification"), 80),
        "keyword_counts": keyword_counts,
        "commit_keyword_counts": commit_info["commit_keyword_counts"],
    }


def validate_project(project: dict[str, Any], row: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    filters = cfg["filters"]
    reasons: list[str] = []

    if filters["drop_forks"] and to_bool(row.get("fork")):
        reasons.append("fork")
    if filters["drop_archived"] and to_bool(row.get("archived")):
        reasons.append("archived")
    if filters["drop_disabled"] and to_bool(row.get("disabled")):
        reasons.append("disabled")
    if to_bool(row.get("private")):
        reasons.append("private")
    if filters["require_language"] and not project["language"]:
        reasons.append("missing_language")
    if filters["require_readme"] and project["readme_chars"] == 0:
        reasons.append("missing_readme")
    if project["readme_chars"] < int(filters["min_readme_chars"]):
        reasons.append("readme_too_short")
    if project["readme_chars"] > int(filters["max_readme_chars"]):
        reasons.append("readme_too_long")
    if project["commit_count"] < int(filters["min_commits"]):
        reasons.append("too_few_commits")

    dur = project.get("duration_hours")
    if dur is None or (isinstance(dur, float) and math.isnan(dur)):
        reasons.append("missing_duration")
    else:
        if dur < float(filters["min_duration_hours"]):
            reasons.append("duration_too_short")
        if dur > float(filters["max_duration_hours"]):
            reasons.append("duration_too_long")
    return reasons


def compute_phase_scores(project: dict[str, Any]) -> dict[str, float]:
    kw = project["keyword_counts"]
    ckw = project["commit_keyword_counts"]

    readme_score = log_norm(project["readme_words"], 2500)
    repo_size_score = log_norm(project["repo_size_kb"], 50000)
    commit_score = log_norm(project["commit_count"], 120)
    contributor_score = log_norm(project["contributors_count"], 8)
    topic_score = log_norm(len(project["topics"]), 12)
    section_score = log_norm(project["markdown_headings"], 20)
    feature_score = log_norm(project["feature_count"], 25)
    external_score = log_norm(kw["external_api"], 12)
    research_score = log_norm(kw["research"] + kw["ml"], 18)
    product_score = log_norm(kw["product"], 20)
    arch_score = log_norm(kw["architecture"], 18)
    test_text_score = log_norm(kw["testing"] + ckw["test"] + ckw["fix"], 20)
    doc_text_score = log_norm(kw["documentation"] + ckw["doc"], 20)
    duration_score = log_norm(project["duration_hours"] or 0.0, 72)

    raw = {
        "research": (
            0.35 * project["scientific_probability"]
            + 0.25 * research_score
            + 0.20 * topic_score
            + 0.10 * readme_score
            + 0.10 * external_score
        ),
        "idea": (
            0.30 * product_score
            + 0.25 * feature_score
            + 0.20 * readme_score
            + 0.15 * topic_score
            + 0.10 * float(project["has_demo"])
        ),
        "selection": (
            0.30 * external_score
            + 0.25 * topic_score
            + 0.20 * arch_score
            + 0.15 * research_score
            + 0.10 * contributor_score
        ),
        "planning": (
            0.30 * arch_score
            + 0.25 * section_score
            + 0.15 * repo_size_score
            + 0.15 * contributor_score
            + 0.15 * float(project["has_install"] or project["has_usage"])
        ),
        "coding": (
            0.35 * repo_size_score
            + 0.30 * commit_score
            + 0.15 * arch_score
            + 0.10 * external_score
            + 0.10 * duration_score
        ),
        "testing": (
            0.30 * test_text_score
            + 0.20 * float(project["has_tests"])
            + 0.20 * commit_score
            + 0.15 * repo_size_score
            + 0.15 * ckw["fix"] / max(1.0, project["commit_count"])
        ),
        "documentation": (
            0.35 * doc_text_score
            + 0.25 * readme_score
            + 0.15 * section_score
            + 0.10 * float(project["has_demo"])
            + 0.10 * float(project["has_install"])
            + 0.05 * float(project["has_license"])
        ),
    }

    return {
        phase: max(0.05, PHASE_PRIORS[phase] * (0.65 + raw[phase]))
        for phase in PHASES
    }


def add_vote(
    votes: list[dict[str, Any]],
    name: str,
    phase: str,
    value: float,
    reason: str,
) -> None:
    value = float(max(-1.0, min(1.0, value)))
    if abs(value) < 1e-6:
        return
    votes.append({"lf": name, "phase": phase, "value": round(value, 4), "reason": reason})


def weak_supervision_votes(project: dict[str, Any]) -> list[dict[str, Any]]:
    """Noisy, interpretable labeling functions for phase workload.

    These rules do not claim to be gold labels. They turn observable HackRep
    project traces into weak evidence about which 0xClaw phase should receive
    more or less budget.
    """

    votes: list[dict[str, Any]] = []
    kw = project["keyword_counts"]
    ckw = project["commit_keyword_counts"]

    repo_size = log_norm(project["repo_size_kb"], 50000)
    commits = log_norm(project["commit_count"], 120)
    duration = log_norm(project["duration_hours"] or 0.0, 72)
    readme = log_norm(project["readme_words"], 2500)
    topics = log_norm(len(project["topics"]), 12)
    contributors = log_norm(project["contributors_count"], 8)
    architecture = log_norm(kw["architecture"], 18)
    external_api = log_norm(kw["external_api"], 12)
    research_terms = log_norm(kw["research"] + kw["ml"], 18)
    product_terms = log_norm(kw["product"], 20)
    testing_terms = log_norm(kw["testing"] + ckw["test"] + ckw["fix"], 20)
    documentation_terms = log_norm(kw["documentation"] + ckw["doc"], 20)
    section_score = log_norm(project["markdown_headings"], 20)
    feature_score = log_norm(project["feature_count"], 25)
    fix_rate = min(1.0, ckw["fix"] / max(1.0, project["commit_count"]))
    feature_rate = min(1.0, ckw["feature"] / max(1.0, project["commit_count"]))

    if repo_size > 0.55 or commits > 0.55:
        add_vote(votes, "LF_code_volume", "coding", 0.45 + 0.35 * max(repo_size, commits), "large repo or many commits")
    elif repo_size < 0.20 and commits < 0.20:
        add_vote(votes, "LF_code_volume", "coding", -0.35, "small repo and few commits")

    if duration > 0.65:
        add_vote(votes, "LF_duration", "coding", 0.25, "long hackathon activity window")
        add_vote(votes, "LF_duration", "testing", 0.20, "long hackathon activity window")
        add_vote(votes, "LF_duration", "planning", 0.15, "long hackathon activity window")
    elif duration < 0.25:
        add_vote(votes, "LF_duration", "research", -0.20, "short activity window")
        add_vote(votes, "LF_duration", "testing", -0.15, "short activity window")

    if project["scientific_probability"] >= 0.55 or research_terms > 0.45:
        strength = 0.35 + 0.35 * max(project["scientific_probability"], research_terms)
        add_vote(votes, "LF_research_scientific", "research", strength, "scientific or ML/research wording")
        add_vote(votes, "LF_research_scientific", "selection", 0.20 + 0.20 * research_terms, "research direction needs careful selection")
    elif project["scientific_probability"] < 0.15 and research_terms < 0.15:
        add_vote(votes, "LF_research_scientific", "research", -0.25, "little scientific/research evidence")

    if product_terms > 0.35 or feature_score > 0.55 or (
        project["has_demo"] and feature_score > 0.25
    ):
        add_vote(votes, "LF_product_surface", "idea", 0.25 + 0.35 * max(product_terms, feature_score), "product/demo/feature evidence")
        add_vote(votes, "LF_product_surface", "documentation", 0.15 + 0.20 * float(project["has_demo"]), "demo-facing project")

    if topics > 0.45 or external_api > 0.35:
        add_vote(votes, "LF_stack_diversity", "selection", 0.30 + 0.35 * max(topics, external_api), "many topics or external APIs")
        add_vote(votes, "LF_stack_diversity", "planning", 0.20 + 0.25 * max(topics, external_api), "integration complexity")
    if architecture > 0.35 or external_api > 0.35:
        add_vote(votes, "LF_integration_risk", "planning", 0.25 + 0.35 * max(architecture, external_api), "architecture/API integration")
        add_vote(votes, "LF_integration_risk", "testing", 0.15 + 0.20 * external_api, "integration needs verification")

    strong_docs = (
        documentation_terms > 0.25
        or project["has_install"]
        or project["has_usage"]
        or (readme > 0.65 and section_score > 0.20)
        or (project["has_demo"] and readme > 0.55)
    )
    if strong_docs:
        add_vote(votes, "LF_docs_readme", "documentation", 0.18 + 0.32 * max(readme, section_score, documentation_terms), "substantial README/setup/usage")
        add_vote(votes, "LF_docs_readme", "planning", 0.10 + 0.15 * section_score, "structured README hints planned components")
    elif project["readme_words"] < 120:
        add_vote(votes, "LF_docs_readme", "documentation", -0.25, "very short README")

    if project["has_tests"] or testing_terms > 0.30:
        add_vote(votes, "LF_testing_signals", "testing", 0.35 + 0.45 * max(testing_terms, float(project["has_tests"])), "tests/CI/fix/bug evidence")
    elif ckw["fix"] == 0 and ckw["test"] == 0 and not project["has_tests"]:
        add_vote(votes, "LF_testing_signals", "testing", -0.25, "little explicit testing evidence")

    if contributors > 0.40:
        add_vote(votes, "LF_team_complexity", "planning", 0.20 + 0.25 * contributors, "multiple contributors")
        add_vote(votes, "LF_team_complexity", "coding", 0.15 + 0.20 * contributors, "multiple contributors")
    elif project["contributors_count"] <= 1:
        add_vote(votes, "LF_team_complexity", "planning", -0.15, "single contributor")

    if fix_rate > 0.05:
        add_vote(votes, "LF_commit_churn", "testing", 0.25 + 0.45 * fix_rate, "fix/debug commit churn")
        add_vote(votes, "LF_commit_churn", "planning", 0.10 + 0.20 * fix_rate, "repair churn suggests planning risk")
    if feature_rate > 0.10:
        add_vote(votes, "LF_commit_churn", "coding", 0.15 + 0.30 * feature_rate, "feature implementation commit churn")

    if external_api > 0.35 and not project["has_tests"]:
        add_vote(votes, "LF_handoff_risk", "testing", 0.25 + 0.25 * external_api, "API-heavy project without explicit tests")
    if architecture > 0.45 and project["readme_words"] < 300:
        add_vote(votes, "LF_handoff_risk", "planning", 0.25, "architecture evidence but sparse docs")

    return votes


def aggregate_weak_votes(project: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    votes = weak_supervision_votes(project)
    by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    for vote in votes:
        by_phase[vote["phase"]].append(vote)

    duration_need = log_norm(project["duration_hours"] or 0.0, 72)
    complexity = float(
        np.mean(
            [
                log_norm(project["repo_size_kb"], 50000),
                log_norm(project["commit_count"], 120),
                log_norm(project["readme_words"], 2500),
                log_norm(len(project["topics"]), 12),
            ]
        )
    )
    base_logit = {
        "research": -0.20,
        "idea": -0.05,
        "selection": -0.05,
        "planning": 0.05,
        "coding": 0.12,
        "testing": 0.02,
        "documentation": -0.05,
    }

    phase_stats: dict[str, dict[str, Any]] = {}
    raw_scores: dict[str, float] = {}
    for phase in PHASES:
        phase_votes = by_phase[phase]
        positive = sum(v["value"] for v in phase_votes if v["value"] > 0)
        negative = -sum(v["value"] for v in phase_votes if v["value"] < 0)
        signed = positive - negative
        total_strength = positive + negative
        conflict = min(positive, negative) / total_strength if total_strength else 0.0
        margin = abs(signed) / total_strength if total_strength else 0.0
        coverage = min(1.0, len(phase_votes) / 4.0)
        confidence = min(
            1.0,
            (0.35 * coverage + 0.65 * (total_strength / (total_strength + 1.2)))
            * (0.55 + 0.45 * margin)
            * (1.0 - 0.50 * conflict),
        )
        score = base_logit[phase] + signed
        raw_scores[phase] = score
        phase_stats[phase] = {
            "phase_votes": phase_votes,
            "positive": positive,
            "negative": negative,
            "signed": signed,
            "total_strength": total_strength,
            "conflict": conflict,
            "margin": margin,
            "confidence": confidence,
        }

    mean_score = sum(raw_scores.values()) / len(raw_scores)

    labels: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        stat = phase_stats[phase]
        phase_votes = stat["phase_votes"]
        centered_score = raw_scores[phase] - mean_score
        # Budget is a relative allocation decision. Centering within each
        # project prevents universal signals, such as every project having a
        # README, from making all phases high-budget at once.
        ratio = 1.0 + 0.42 * math.tanh(centered_score / 0.95)
        ratio = min(1.25, max(0.50, ratio))
        time_need = (
            0.50 * ratio
            + 0.20 * duration_need
            + 0.15 * complexity
            + 0.15 * PHASE_TIME_PRIORS[phase]
        )

        token_label = ratio_to_bin(ratio, cfg["budget_bins"]["token"])
        time_label = ratio_to_bin(time_need, cfg["budget_bins"]["time"])

        if phase == "documentation" and ratio <= 0.875:
            think = "THINK_OFF"
        elif phase in {"selection", "testing", "research"} and ratio >= 1.125:
            think = "THINK_HIGH"
        elif ratio >= 0.875:
            think = "THINK_LOW"
        else:
            think = "THINK_OFF"

        if phase == "testing":
            repair = 2 if ratio >= 1.125 else (1 if ratio >= 0.875 else 0)
        elif phase == "coding":
            repair = 2 if ratio >= 1.125 else (1 if ratio >= 0.875 else 0)
        elif phase in {"selection", "planning"}:
            repair = 1 if ratio >= 1.125 else 0
        else:
            repair = 0

        labels[phase] = {
            "phase_score": round(score, 4),
            "relative_token_ratio": round(ratio, 4),
            "time_need_score": round(float(time_need), 4),
            "token_label": token_label,
            "time_label": time_label,
            "think_label": think,
            "repair_rounds": repair,
            "weak_confidence": round(float(stat["confidence"]), 4),
            "weak_confidence_bin": confidence_bin(float(stat["confidence"])),
            "weak_margin": round(float(stat["margin"]), 4),
            "weak_conflict": round(float(stat["conflict"]), 4),
            "weak_vote_count": len(phase_votes),
            "weak_positive": round(float(stat["positive"]), 4),
            "weak_negative": round(float(stat["negative"]), 4),
            "weak_raw_score": round(float(raw_scores[phase]), 4),
            "weak_centered_score": round(float(centered_score), 4),
            "weak_lfs_positive": [
                v["lf"] for v in phase_votes if v["value"] > 0
            ][:8],
            "weak_lfs_negative": [
                v["lf"] for v in phase_votes if v["value"] < 0
            ][:8],
        }
    return labels


def phase_labels(project: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if cfg.get("weak_supervision", {}).get("enabled", True):
        return aggregate_weak_votes(project, cfg)

    scores = compute_phase_scores(project)
    mean_score = sum(scores.values()) / len(scores)
    duration_need = log_norm(project["duration_hours"] or 0.0, 72)
    complexity = np.mean(
        [
            log_norm(project["repo_size_kb"], 50000),
            log_norm(project["commit_count"], 120),
            log_norm(project["readme_words"], 2500),
            log_norm(len(project["topics"]), 12),
        ]
    )

    labels: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        ratio = scores[phase] / max(mean_score, 1e-6)

        if phase == "coding":
            kw = project["keyword_counts"]
            coding_complexity = (
                0.45 * log_norm(project["repo_size_kb"], 50000)
                + 0.30 * log_norm(project["commit_count"], 120)
                + 0.10 * log_norm(kw["architecture"], 18)
                + 0.10 * log_norm(kw["external_api"], 12)
                + 0.05 * log_norm(project["contributors_count"], 8)
            )
            ratio = 0.72 + 0.48 * coding_complexity + 0.08 * duration_need

        elif phase == "planning":
            ratio = 0.35 * ratio + 0.65 * (
                0.78
                + 0.30 * log_norm(project["markdown_headings"], 20)
                + 0.18 * log_norm(project["repo_size_kb"], 50000)
                + 0.10 * log_norm(project["contributors_count"], 8)
            )

        elif phase == "research":
            ratio = 0.45 * ratio + 0.55 * (
                0.62
                + 0.30 * project["scientific_probability"]
                + 0.18 * log_norm(
                    project["keyword_counts"]["research"] + project["keyword_counts"]["ml"],
                    18,
                )
            )

        ratio = min(1.25, max(0.25, ratio))
        time_need = (
            0.50 * ratio
            + 0.20 * duration_need
            + 0.15 * complexity
            + 0.15 * PHASE_TIME_PRIORS[phase]
        )

        token_label = ratio_to_bin(ratio, cfg["budget_bins"]["token"])
        time_label = ratio_to_bin(time_need, cfg["budget_bins"]["time"])

        if phase == "documentation" and ratio <= 0.625:
            think = "THINK_OFF"
        elif phase in {"selection", "testing", "research"} and ratio >= 1.0:
            think = "THINK_HIGH"
        elif ratio >= 0.75:
            think = "THINK_LOW"
        else:
            think = "THINK_OFF"

        if phase == "testing":
            repair = 2 if ratio >= 0.875 else 1
        elif phase == "coding":
            repair = 1 if ratio < 1.125 else 2
        elif phase == "selection":
            repair = 1 if ratio >= 0.875 else 0
        else:
            repair = 0

        labels[phase] = {
            "phase_score": round(scores[phase], 4),
            "relative_token_ratio": round(ratio, 4),
            "time_need_score": round(float(time_need), 4),
            "token_label": token_label,
            "time_label": time_label,
            "think_label": think,
            "repair_rounds": repair,
        }
    return labels


def make_prompt(project: dict[str, Any], phase: str, labels: dict[str, Any]) -> str:
    compact_features = {
        "language": project["language"],
        "topics": project["topics"][:12],
        "duration_bin": project["duration_bin"],
        "duration_hours": round(project["duration_hours"] or 0.0, 2),
        "repo_size_kb": project["repo_size_kb"],
        "readme_words": project["readme_words"],
        "commit_count": project["commit_count"],
        "contributors_count": project["contributors_count"],
        "has_demo": project["has_demo"],
        "has_install": project["has_install"],
        "has_usage": project["has_usage"],
        "has_tests": project["has_tests"],
        "scientific_probability": round(project["scientific_probability"], 3),
        "keyword_counts": project["keyword_counts"],
        "commit_keyword_counts": project["commit_keyword_counts"],
    }
    return (
        "Allocate the next-phase budget for a hackathon agent.\n"
        "Return only control tokens.\n\n"
        f"Project: {project['full_name']}\n"
        f"Description: {project['description']}\n"
        f"Phase: {phase}\n"
        f"Project features: {json.dumps(compact_features, ensure_ascii=False, sort_keys=True)}\n"
        f"README excerpt: {project['readme_excerpt']}\n"
    )


def make_target(phase: str, label: dict[str, Any]) -> str:
    return (
        f"<PHASE_{phase.upper()}>"
        f"<{label['token_label']}>"
        f"<{label['time_label']}>"
        f"<{label['think_label']}>"
        f"<REPAIR_{label['repair_rounds']}>"
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=json_default)


def extract_parquet_from_zip(zip_path: Path, parquet_name: str, work_dir: Path) -> Path:
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / parquet_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            info = zf.getinfo(parquet_name)
        except KeyError as exc:
            names = ", ".join(zf.namelist()[:20])
            raise FileNotFoundError(
                f"{parquet_name} not found in {zip_path}. First entries: {names}"
            ) from exc
        with zf.open(info) as src, out_path.open("wb") as dst:
            with tqdm(total=info.file_size, unit="B", unit_scale=True, desc="Extract parquet") as bar:
                while True:
                    chunk = src.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    dst.write(chunk)
                    bar.update(len(chunk))
    return out_path


def open_outputs(output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    handles["projects"] = (output_dir / cfg["output"]["project_jsonl"]).open(
        "w", encoding="utf-8"
    )
    for split in ("train", "val", "test"):
        handles[f"sft_{split}"] = (
            output_dir / f"{cfg['output']['sft_prefix']}_{split}.jsonl"
        ).open("w", encoding="utf-8")
        handles[f"flat_{split}"] = (
            output_dir / f"{cfg['output']['flat_prefix']}_{split}.jsonl"
        ).open("w", encoding="utf-8")
    return handles


def close_outputs(handles: dict[str, Any]) -> None:
    for handle in handles.values():
        handle.close()


def process_dataset(parquet_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(cfg["output"]["output_dir"])
    handles = open_outputs(output_dir, cfg)

    stats = Counter()
    invalid_reasons = Counter()
    split_counts = Counter()
    phase_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    phase_time_counts: dict[str, Counter[str]] = defaultdict(Counter)
    phase_confidence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lf_positive_counts = Counter()
    lf_negative_counts = Counter()

    parquet_file = pq.ParquetFile(parquet_path)
    available_cols = set(parquet_file.schema_arrow.names)
    columns = [col for col in SELECTED_COLUMNS if col in available_cols]
    missing_cols = sorted(set(SELECTED_COLUMNS) - available_cols)

    limit = cfg["processing"].get("limit")
    batch_size = int(cfg["processing"]["batch_size"])
    keep_invalid = bool(cfg["processing"]["keep_invalid_projects"])
    processed_valid = 0

    try:
        row_iter = parquet_file.iter_batches(batch_size=batch_size, columns=columns)
        total_batches = math.ceil(parquet_file.metadata.num_rows / batch_size)
        for batch in tqdm(row_iter, total=total_batches, desc="Read batches"):
            df = batch.to_pandas()
            for row in df.to_dict(orient="records"):
                stats["rows_seen"] += 1
                project = project_features(row, cfg)
                reasons = validate_project(project, row, cfg)
                project["valid"] = len(reasons) == 0
                project["filter_reasons"] = reasons
                project["split"] = split_for_name(project["full_name"], cfg)

                if reasons:
                    invalid_reasons.update(reasons)
                    stats["projects_invalid"] += 1
                    if not keep_invalid:
                        continue

                labels = phase_labels(project, cfg)
                ws_cfg = cfg.get("weak_supervision", {})
                if ws_cfg.get("drop_low_confidence", False):
                    min_conf = float(ws_cfg.get("min_phase_confidence", 0.0))
                    labels = {
                        phase: label
                        for phase, label in labels.items()
                        if float(label.get("weak_confidence", 1.0)) >= min_conf
                    }
                    if not labels:
                        stats["projects_without_confident_labels"] += 1
                        continue
                project["phase_budget_labels"] = labels
                handles["projects"].write(dumps(project) + "\n")

                split = project["split"]
                split_counts[split] += 1
                stats["projects_written"] += 1
                processed_valid += 1

                for phase, label in labels.items():
                    phase_token_counts[phase][label["token_label"]] += 1
                    phase_time_counts[phase][label["time_label"]] += 1
                    phase_confidence_counts[phase][
                        label.get("weak_confidence_bin", "unknown")
                    ] += 1
                    lf_positive_counts.update(label.get("weak_lfs_positive", []))
                    lf_negative_counts.update(label.get("weak_lfs_negative", []))

                    user_prompt = make_prompt(project, phase, label)
                    target = make_target(phase, label)
                    sft = {
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a phase-aware budget planner for a hackathon agent. "
                                    "Choose compact control tokens for the requested phase."
                                ),
                            },
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": target},
                        ],
                        "metadata": {
                            "project_id": project["full_name"],
                            "phase": phase,
                            "split": split,
                            **label,
                        },
                    }
                    flat = {
                        "project_id": project["full_name"],
                        "phase": phase,
                        "split": split,
                        "input": user_prompt,
                        "target": target,
                        "label": label,
                        "features": {
                            key: project[key]
                            for key in [
                                "language",
                                "duration_hours",
                                "duration_bin",
                                "repo_size_kb",
                                "readme_words",
                                "commit_count",
                                "contributors_count",
                                "scientific_probability",
                                "keyword_counts",
                                "commit_keyword_counts",
                            ]
                        },
                    }
                    handles[f"sft_{split}"].write(dumps(sft) + "\n")
                    handles[f"flat_{split}"].write(dumps(flat) + "\n")

                if limit and processed_valid >= int(limit):
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        close_outputs(handles)

    manifest = {
        "parquet_path": str(parquet_path),
        "output_dir": str(output_dir),
        "selected_columns": columns,
        "missing_columns": missing_cols,
        "stats": dict(stats),
        "invalid_reasons": dict(invalid_reasons.most_common()),
        "split_counts": dict(split_counts),
        "phase_token_counts": {
            phase: dict(counter) for phase, counter in phase_token_counts.items()
        },
        "phase_time_counts": {
            phase: dict(counter) for phase, counter in phase_time_counts.items()
        },
        "phase_confidence_counts": {
            phase: dict(counter) for phase, counter in phase_confidence_counts.items()
        },
        "lf_positive_counts": dict(lf_positive_counts.most_common()),
        "lf_negative_counts": dict(lf_negative_counts.most_common()),
        "config": cfg,
    }
    manifest_path = output_dir / cfg["output"]["manifest"]
    manifest_path.write_text(dumps(manifest), encoding="utf-8")
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HackRep phase-budget SFT data for a 0xClaw budget policy model."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument("--zip-path", type=Path, default=None, help="Path to 17572684.zip.")
    parser.add_argument("--parquet-path", type=Path, default=None, help="Existing HackRep.parquet path.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working/cache directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit valid projects for smoke tests.")
    parser.add_argument("--keep-invalid-projects", action="store_true")
    parser.add_argument("--train-pct", type=int, default=None, help="Project-level train split percentage.")
    parser.add_argument("--val-pct", type=int, default=None, help="Project-level validation split percentage.")
    parser.add_argument("--test-pct", type=int, default=None, help="Project-level test split percentage.")
    parser.add_argument(
        "--min-phase-confidence",
        type=float,
        default=None,
        help="Minimum weak-supervision confidence when --drop-low-confidence is set.",
    )
    parser.add_argument(
        "--drop-low-confidence",
        action="store_true",
        help="Drop phase samples with weak-supervision confidence below --min-phase-confidence.",
    )
    parser.add_argument(
        "--disable-weak-supervision",
        action="store_true",
        help="Use the older monolithic heuristic labeler instead of weak supervision.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Require --parquet-path; do not extract zip.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = deep_update(default_config(), read_yaml(args.config))

    if args.zip_path:
        cfg["input"]["zip_path"] = str(args.zip_path)
    if args.work_dir:
        cfg["input"]["work_dir"] = str(args.work_dir)
    if args.output_dir:
        cfg["output"]["output_dir"] = str(args.output_dir)
    if args.batch_size:
        cfg["processing"]["batch_size"] = args.batch_size
    if args.limit:
        cfg["processing"]["limit"] = args.limit
    if args.keep_invalid_projects:
        cfg["processing"]["keep_invalid_projects"] = True
    if args.train_pct is not None:
        cfg["splits"]["train_pct"] = args.train_pct
    if args.val_pct is not None:
        cfg["splits"]["val_pct"] = args.val_pct
    if args.test_pct is not None:
        cfg["splits"]["test_pct"] = args.test_pct
    split_sum = int(cfg["splits"]["train_pct"]) + int(cfg["splits"]["val_pct"]) + int(cfg["splits"]["test_pct"])
    if split_sum != 100:
        raise SystemExit(f"Split percentages must sum to 100, got {split_sum}.")
    if args.min_phase_confidence is not None:
        cfg["weak_supervision"]["min_phase_confidence"] = args.min_phase_confidence
    if args.drop_low_confidence:
        cfg["weak_supervision"]["drop_low_confidence"] = True
    if args.disable_weak_supervision:
        cfg["weak_supervision"]["enabled"] = False

    work_dir = Path(cfg["input"]["work_dir"])
    parquet_name = cfg["input"]["parquet_name"]
    if args.parquet_path:
        parquet_path = args.parquet_path
    else:
        if args.no_extract:
            raise SystemExit("--no-extract requires --parquet-path")
        zip_path = Path(cfg["input"]["zip_path"])
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        parquet_path = extract_parquet_from_zip(zip_path, parquet_name, work_dir)

    manifest = process_dataset(parquet_path, cfg)
    print(json.dumps(manifest["stats"], indent=2, ensure_ascii=False))
    print(f"Wrote outputs to: {manifest['output_dir']}")
    print(f"Manifest: {Path(manifest['output_dir']) / cfg['output']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
