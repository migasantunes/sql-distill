"""Deterministic teacher for tests and dry runs. No network, no randomness."""

from __future__ import annotations

from typing import Any


class FakeTeacher:
    """Returns the first ``n`` of a fixed list of pairs and records every call."""

    name = "fake"

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = list(pairs)
        self.calls: list[dict[str, Any]] = []

    def generate(self, schema_text: str, *, n: int, difficulty: str) -> list[tuple[str, str]]:
        self.calls.append({"schema_text": schema_text, "n": n, "difficulty": difficulty})
        # No cycling: asking for more than exists returns what exists.
        return list(self.pairs[: max(n, 0)])
