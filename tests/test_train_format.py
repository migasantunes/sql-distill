from __future__ import annotations

from sqldistill import prompts
from sqldistill.records import Example
from sqldistill.train.format import to_chat_example

SCHEMA = "CREATE TABLE singer (\n  singer_id INTEGER PRIMARY KEY,\n  name TEXT\n);\n"


def test_prompt_part_is_student_messages_exactly() -> None:
    # Invariant 3: training uses the very same prompt as baseline and eval.
    ex = Example(
        id="t1",
        db_id="toy",
        question="List names.",
        sql="SELECT name FROM singer",
        source="teacher",
    )
    out = to_chat_example(ex, SCHEMA)
    assert list(out) == ["messages"]
    expected = prompts.student_messages(SCHEMA, ex.question, ex.evidence)
    assert out["messages"][:-1] == expected
    assert out["messages"][-1] == {"role": "assistant", "content": "SELECT name FROM singer"}


def test_evidence_is_passed_through() -> None:
    ex = Example(
        id="t2",
        db_id="toy",
        question="Q?",
        sql="SELECT 1",
        source="gold",
        evidence="adult means age >= 18",
    )
    out = to_chat_example(ex, SCHEMA)
    assert out["messages"][:-1] == prompts.student_messages(SCHEMA, "Q?", "adult means age >= 18")
