"""JSONL record types shared by every stage."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class Example(BaseModel):
    """One question/SQL pair. The unit every stage exchanges."""

    id: str
    db_id: str
    question: str
    sql: str
    source: Literal["teacher", "gold"]
    evidence: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    """One model prediction for one example."""

    example_id: str
    db_id: str
    predicted_sql: str
    raw_output: str
    latency_s: float | None = None


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    """Read a JSONL file, validating every non-blank line as ``model``."""
    rows: list[T] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(model.model_validate_json(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    """Write one JSON object per line (UTF-8), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(row.model_dump_json())
            fh.write("\n")
