"""Ask a teacher for candidate question/SQL pairs over a set of databases."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sqldistill.prompts import DIFFICULTIES
from sqldistill.records import Example
from sqldistill.schema import introspect, render_schema, sample_rows
from sqldistill.teacher.base import Teacher

SAMPLE_ROWS_PER_TABLE = 3


def split_evenly(total: int, k: int) -> list[int]:
    """Split ``total`` into ``k`` near-equal parts; the remainder goes to the first parts."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def generate_candidates(
    db_paths: dict[str, Path],
    teacher: Teacher,
    *,
    per_db: int,
    difficulties: Sequence[str],
) -> list[Example]:
    """Generate up to ``per_db`` candidates per database, spread over ``difficulties``.

    Databases are visited in sorted ``db_id`` order and ids are
    ``f"{db_id}-{difficulty}-{i:04d}"``, so the same teacher output always gives
    the same file. Candidates are unverified: filtering is a separate stage.
    """
    if per_db < 1:
        raise ValueError(f"per_db must be >= 1, got {per_db}")
    difficulties = list(difficulties)
    if not difficulties:
        raise ValueError("difficulties must not be empty")
    unknown = [d for d in difficulties if d not in DIFFICULTIES]
    if unknown:
        raise ValueError(f"unknown difficulties {unknown}; expected some of {list(DIFFICULTIES)}")
    if len(set(difficulties)) != len(difficulties):
        raise ValueError(f"duplicate difficulties would give duplicate ids: {difficulties}")

    counts = split_evenly(per_db, len(difficulties))
    examples: list[Example] = []
    for db_id in sorted(db_paths):
        path = Path(db_paths[db_id])
        schema = introspect(path, db_id=db_id)
        schema_text = render_schema(
            schema, sample_rows=sample_rows(path, schema, n=SAMPLE_ROWS_PER_TABLE)
        )
        for difficulty, n in zip(difficulties, counts, strict=True):
            if n == 0:
                continue
            pairs = teacher.generate(schema_text, n=n, difficulty=difficulty)
            for i, (question, sql) in enumerate(pairs):
                examples.append(
                    Example(
                        id=f"{db_id}-{difficulty}-{i:04d}",
                        db_id=db_id,
                        question=question,
                        sql=sql,
                        source="teacher",
                        meta={"teacher": teacher.name, "difficulty": difficulty},
                    )
                )
    return examples
