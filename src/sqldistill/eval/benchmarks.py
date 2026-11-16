"""Benchmark loaders: Spider and BIRD Mini-Dev as lists of gold Examples.

The expected on-disk layouts below follow the public releases as documented,
but they MUST be re-verified against the real downloads in M1 (docs/PLAN.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqldistill.records import Example

_SPIDER_FILES = {"dev": "dev.json", "train": "train_spider.json"}
_BIRD_FILE = "mini_dev_sqlite.json"


@dataclass(frozen=True)
class Benchmark:
    name: str
    examples: list[Example]
    db_paths: dict[str, Path]


def _read_items(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. Download the benchmark first (docs/PLAN.md, M1)."
        )
    with path.open(encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a JSON list of examples, got {type(items).__name__}")
    return items


def _resolve_dbs(db_root: Path, db_ids: list[str]) -> dict[str, Path]:
    """``db_root/<db_id>/<db_id>.sqlite`` for every distinct db_id, in first-seen order."""
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for db_id in db_ids:
        if db_id in paths:
            continue
        path = db_root / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            missing.append(str(path))
        paths[db_id] = path
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise FileNotFoundError(
            f"{len(missing)} referenced database(s) missing: {shown}. "
            "Download the benchmark databases first (docs/PLAN.md, M1)."
        )
    return paths


def load_spider(root: Path, split: str = "dev") -> Benchmark:
    """Load Spider ``split`` ("dev" or "train") from ``root``.

    Expected layout (standard Spider release; re-verify against the real
    download in M1)::

        root/dev.json            # or train_spider.json for split="train"
        root/database/<db_id>/<db_id>.sqlite

    Each JSON item needs "db_id", "question" and "query".
    """
    if split not in _SPIDER_FILES:
        raise ValueError(f"unknown Spider split {split!r}; expected one of {sorted(_SPIDER_FILES)}")
    root = Path(root)
    name = f"spider_{split}"
    items = _read_items(root / _SPIDER_FILES[split])
    examples = [
        Example(
            id=f"{name}-{i:05d}",
            db_id=item["db_id"],
            question=item["question"],
            sql=item["query"],
            source="gold",
        )
        for i, item in enumerate(items)
    ]
    db_paths = _resolve_dbs(root / "database", [ex.db_id for ex in examples])
    return Benchmark(name=name, examples=examples, db_paths=db_paths)


def load_bird_minidev(root: Path) -> Benchmark:
    """Load BIRD Mini-Dev (SQLite version) from ``root``.

    Expected layout (re-verify against the real download in M1)::

        root/mini_dev_sqlite.json
        root/dev_databases/<db_id>/<db_id>.sqlite

    Each JSON item needs "db_id", "question", "evidence" and "SQL"; "question_id"
    and "difficulty" are kept in ``meta`` when present.
    """
    root = Path(root)
    name = "bird_minidev"
    items = _read_items(root / _BIRD_FILE)
    examples = []
    for i, item in enumerate(items):
        meta = {k: item[k] for k in ("question_id", "difficulty") if k in item}
        evidence = item.get("evidence") or None  # "" means no hint
        examples.append(
            Example(
                id=f"{name}-{i:05d}",
                db_id=item["db_id"],
                question=item["question"],
                sql=item["SQL"],
                source="gold",
                evidence=evidence,
                meta=meta,
            )
        )
    db_paths = _resolve_dbs(root / "dev_databases", [ex.db_id for ex in examples])
    return Benchmark(name=name, examples=examples, db_paths=db_paths)
