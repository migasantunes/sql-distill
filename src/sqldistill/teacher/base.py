"""The Teacher protocol shared by every teacher backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Teacher(Protocol):
    """Writes ``n`` (question, sql) pairs for a rendered schema at a difficulty."""

    name: str

    def generate(self, schema_text: str, *, n: int, difficulty: str) -> list[tuple[str, str]]: ...
