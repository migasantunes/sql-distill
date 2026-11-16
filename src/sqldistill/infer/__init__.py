"""Inference: the SQLGenerator protocol, SQL extraction and the HF backend (M3)."""

from __future__ import annotations

from sqldistill.infer.base import SQLGenerator
from sqldistill.infer.extract import extract_sql

__all__ = ["SQLGenerator", "extract_sql"]
