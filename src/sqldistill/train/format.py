"""Turn an Example into a chat-format training record.

The prompt part comes from ``prompts.student_messages`` -- the same function
the baseline and evaluation use -- so the training format cannot drift from the
evaluation format (invariant 3).
"""

from __future__ import annotations

from sqldistill import prompts
from sqldistill.records import Example


def to_chat_example(example: Example, schema_text: str) -> dict[str, list[dict[str, str]]]:
    """``{"messages": student_messages(...) + [assistant turn with the SQL]}``."""
    messages = prompts.student_messages(schema_text, example.question, example.evidence)
    messages.append({"role": "assistant", "content": example.sql})
    return {"messages": messages}
