"""Hugging Face transformers backend for the student model (M3).

``torch`` and ``transformers`` are imported inside methods only, so importing
this module stays cheap and works with the core install.
"""

from __future__ import annotations

from pathlib import Path


class HFGenerator:
    """Local student model (optionally with a LoRA adapter) behind SQLGenerator."""

    def __init__(
        self,
        model_id: str,
        adapter_path: Path | None = None,
        max_new_tokens: int = 256,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.name = model_id + ("+adapter" if adapter_path is not None else "")

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Greedy-decode a reply to ``messages`` and return the raw text."""
        raise NotImplementedError(
            "M3: HFGenerator.generate (4-bit load, chat template, greedy decoding, batching)"
        )
