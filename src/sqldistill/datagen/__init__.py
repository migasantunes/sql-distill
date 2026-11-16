"""Synthetic data generation: teacher candidates, execution filter, dedup."""

from __future__ import annotations

from sqldistill.datagen.dedup import dedup, normalize_sql
from sqldistill.datagen.filter import FilterLevel, FilterReport, filter_examples
from sqldistill.datagen.generate import generate_candidates

__all__ = [
    "FilterLevel",
    "FilterReport",
    "dedup",
    "filter_examples",
    "generate_candidates",
    "normalize_sql",
]
