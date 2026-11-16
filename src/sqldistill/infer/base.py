"""The interface every SQL-producing model implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SQLGenerator(Protocol):
    """Anything that turns chat messages into raw model text.

    ``generate`` returns the untouched model output; SQL extraction is done
    separately by :func:`sqldistill.infer.extract.extract_sql`, so the raw text
    can be stored and the extraction step audited on its own.
    """

    name: str

    def generate(self, messages: list[dict[str, str]]) -> str: ...
