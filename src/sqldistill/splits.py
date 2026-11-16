"""Leakage guard: train and eval databases must be disjoint."""

from __future__ import annotations

from collections.abc import Iterable

from sqldistill.records import Example


class LeakageError(ValueError):
    """Raised when a database appears in both training and evaluation data."""


def assert_disjoint(train_db_ids: Iterable[str], eval_db_ids: Iterable[str]) -> None:
    """Raise LeakageError if any db_id is shared between train and eval."""
    overlap = set(train_db_ids) & set(eval_db_ids)
    if overlap:
        ids = ", ".join(sorted(overlap))
        raise LeakageError(f"{len(overlap)} database(s) in both train and eval: {ids}")


def db_ids_of(examples: Iterable[Example]) -> set[str]:
    """The set of db_ids appearing in ``examples``."""
    return {ex.db_id for ex in examples}
