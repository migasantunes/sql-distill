"""API-backed teacher (M2). ``anthropic`` is imported lazily inside methods,
so this module imports fine with only core dependencies installed."""

from __future__ import annotations


class AnthropicTeacher:
    """Teacher backed by the Anthropic Messages API. Implemented in M2."""

    def __init__(self, model: str, temperature: float = 0.8, max_tokens: int = 4096):
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, schema_text: str, *, n: int, difficulty: str) -> list[tuple[str, str]]:
        raise NotImplementedError(
            "M2: implement AnthropicTeacher with retries, on-disk cache and cost log"
        )
