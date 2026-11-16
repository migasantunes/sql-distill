"""All prompt templates: the student prompt and the teacher generation prompt.

``student_messages`` is the one and only student prompt format. Training
targets, the zero-shot baseline and evaluation all go through it, so any change
here changes the experiment (invariant 3). ``tests/test_prompts.py`` pins its
exact output.
"""

from __future__ import annotations

import json
import re
from typing import Any

STUDENT_SYSTEM = (
    "You are a SQLite expert. Given a database schema and a question, "
    "answer with a single SQLite query and nothing else."
)

# One-line definition of each difficulty tier, shown to the teacher.
DIFFICULTIES: dict[str, str] = {
    "simple": "a single table, with filtering, sorting or a simple aggregate",
    "moderate": "joins between tables and/or aggregation with GROUP BY",
    "challenging": (
        "nested subqueries, set operations (UNION / INTERSECT / EXCEPT), "
        "or multiple joins combined with grouping"
    ),
}


def student_messages(
    schema_text: str, question: str, evidence: str | None = None
) -> list[dict[str, str]]:
    """Build the [system, user] chat messages for the student.

    The user turn is the schema, then an optional ``-- Hint:`` line (BIRD
    evidence), then the question. Blank or missing evidence adds no hint line.
    """
    parts = [schema_text.rstrip()]
    if evidence is not None and evidence.strip():
        parts.append(f"-- Hint: {evidence.strip()}")
    parts.append(f"Question: {question.strip()}")
    return [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def teacher_generation_prompt(schema_text: str, n: int, difficulty: str) -> str:
    """Prompt asking the teacher for ``n`` question/SQL pairs at ``difficulty``."""
    if difficulty not in DIFFICULTIES:
        known = ", ".join(DIFFICULTIES)
        raise ValueError(f"unknown difficulty {difficulty!r}; expected one of: {known}")
    tiers = "\n".join(f"- {name}: {desc}" for name, desc in DIFFICULTIES.items())
    return (
        "You are writing training data for a text-to-SQL model.\n\n"
        "Here is a SQLite database schema, with a few sample rows per table:\n\n"
        f"{schema_text.rstrip()}\n\n"
        f"Write {n} diverse pairs of a natural-language question and the SQLite query "
        f"that answers it. Every query must be at the {difficulty!r} difficulty level.\n\n"
        f"Difficulty levels:\n{tiers}\n\n"
        "Rules:\n"
        "- Use only the tables and columns that appear in the schema above.\n"
        "- Write questions the way a real user of this database would ask them; "
        "avoid naming columns or tables verbatim where a natural phrasing exists.\n"
        "- Each query must be valid SQLite and answer its question exactly.\n"
        "- Vary the questions: do not ask the same thing twice in different words.\n\n"
        'Answer with a JSON array of objects with exactly the keys "question" and "sql", '
        "and nothing else."
    )


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*\s*\n?(.*?)```", re.DOTALL)


def _first_json_array(text: str) -> list[Any] | None:
    """Return the first JSON array that can be decoded from ``text``, or None."""
    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            pass
        else:
            if isinstance(value, list):
                return value
        start = text.find("[", start + 1)
    return None


def parse_teacher_response(text: str) -> list[tuple[str, str]]:
    """Extract (question, sql) pairs from a teacher response. Never raises.

    Tolerates code fences and prose around the JSON. Items that are not objects
    with non-empty string ``question`` and ``sql`` are skipped; anything
    unparseable gives ``[]``.
    """
    if not isinstance(text, str):
        return []
    fenced = _FENCE_RE.search(text)
    candidates = [fenced.group(1), text] if fenced else [text]
    items: list[Any] | None = None
    for candidate in candidates:
        items = _first_json_array(candidate)
        if items is not None:
            break
    if items is None:
        return []
    pairs: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question, sql = item.get("question"), item.get("sql")
        if not isinstance(question, str) or not isinstance(sql, str):
            continue
        question, sql = question.strip(), sql.strip()
        if question and sql:
            pairs.append((question, sql))
    return pairs
