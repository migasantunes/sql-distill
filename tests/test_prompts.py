from __future__ import annotations

import pytest

from sqldistill.prompts import (
    STUDENT_SYSTEM,
    parse_teacher_response,
    student_messages,
    teacher_generation_prompt,
)

SCHEMA = "CREATE TABLE t (\n  a INTEGER\n);\n"


def test_student_messages_golden() -> None:
    # Pins the exact student prompt: train, baseline and eval all use it (invariant 3).
    assert student_messages(SCHEMA, "How many rows?") == [
        {
            "role": "system",
            "content": "You are a SQLite expert. Given a database schema and a question, "
            "answer with a single SQLite query and nothing else.",
        },
        {
            "role": "user",
            "content": "CREATE TABLE t (\n  a INTEGER\n);\n\nQuestion: How many rows?",
        },
    ]


def test_student_messages_golden_with_evidence() -> None:
    messages = student_messages(SCHEMA, "How many rows?", evidence="rows means records")
    assert messages[0]["content"] == STUDENT_SYSTEM
    assert messages[1]["content"] == (
        "CREATE TABLE t (\n  a INTEGER\n);\n\n"
        "-- Hint: rows means records\n\n"
        "Question: How many rows?"
    )


@pytest.mark.parametrize("evidence", [None, "", "   "])
def test_student_messages_no_hint_without_evidence(evidence: str | None) -> None:
    content = student_messages(SCHEMA, "Q?", evidence=evidence)[1]["content"]
    assert "Hint" not in content


def test_student_messages_is_deterministic() -> None:
    assert student_messages(SCHEMA, "Q?", "e") == student_messages(SCHEMA, "Q?", "e")


@pytest.mark.parametrize("difficulty", ["simple", "moderate", "challenging"])
def test_teacher_prompt_contents(difficulty: str) -> None:
    prompt = teacher_generation_prompt(SCHEMA, 7, difficulty)
    assert SCHEMA.rstrip() in prompt
    assert "Write 7 " in prompt
    assert repr(difficulty) in prompt
    assert '"question"' in prompt and '"sql"' in prompt
    assert "JSON array" in prompt
    for tier in ("simple:", "moderate:", "challenging:"):
        assert tier in prompt


def test_teacher_prompt_unknown_difficulty() -> None:
    with pytest.raises(ValueError, match="unknown difficulty"):
        teacher_generation_prompt(SCHEMA, 3, "impossible")


def test_parse_plain_array() -> None:
    text = '[{"question": " How many? ", "sql": " SELECT 1 "}]'
    assert parse_teacher_response(text) == [("How many?", "SELECT 1")]


def test_parse_fenced_with_prose() -> None:
    text = (
        "Sure! Here are the pairs:\n```json\n"
        '[{"question": "Q1", "sql": "SELECT 1"},\n {"question": "Q2", "sql": "SELECT 2"}]\n'
        "```\nHope this helps [really]."
    )
    assert parse_teacher_response(text) == [("Q1", "SELECT 1"), ("Q2", "SELECT 2")]


def test_parse_skips_bad_items() -> None:
    text = (
        '[{"question": "ok", "sql": "SELECT 1"}, {"question": "", "sql": "SELECT 2"},'
        ' {"question": "no sql"}, {"question": 3, "sql": "SELECT 3"}, "junk", [1],'
        ' {"question": "q", "sql": "   "}, {"question": "extra", "sql": "SELECT 4", "x": 1}]'
    )
    assert parse_teacher_response(text) == [("ok", "SELECT 1"), ("extra", "SELECT 4")]


def test_parse_skips_non_json_brackets_before_array() -> None:
    text = 'Pairs [see below]: [{"question": "Q", "sql": "SELECT 1"}]'
    assert parse_teacher_response(text) == [("Q", "SELECT 1")]


@pytest.mark.parametrize(
    "text",
    ["", "no json here", "[", '{"question": "Q", "sql": "S"}', "```json\n[oops\n```", None, 42],
)
def test_parse_never_raises(text: object) -> None:
    assert parse_teacher_response(text) == []  # type: ignore[arg-type]
