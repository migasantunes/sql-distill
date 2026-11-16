from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sqldistill.records import Example, Prediction, read_jsonl, write_jsonl


def _example(i: int, **kw) -> Example:
    return Example(
        id=f"ex{i}",
        db_id="toy",
        question=f"Question {i}?",
        sql="SELECT 1",
        source="teacher",
        **kw,
    )


def test_meta_default_is_not_shared() -> None:
    a, b = _example(1), _example(2)
    a.meta["k"] = 1
    assert b.meta == {}


def test_source_is_restricted() -> None:
    with pytest.raises(ValidationError):
        Example(id="x", db_id="d", question="q", sql="s", source="human")


def test_round_trip_preserves_non_ascii(tmp_path: Path) -> None:
    rows = [
        _example(1, meta={"temperature": 0.8, "model": "t"}),
        _example(2, evidence="Chloé is a singer"),
        Example(
            id="ex3",
            db_id="toy",
            question="Which singers are from France, like Chloé?",
            sql="SELECT name FROM singer WHERE name = 'Chloé'",
            source="gold",
        ),
    ]
    path = tmp_path / "nested" / "dir" / "examples.jsonl"
    write_jsonl(path, rows)

    assert read_jsonl(path, Example) == rows
    text = path.read_text(encoding="utf-8")
    assert "Chloé" in text  # stored readably, not as \u escapes
    assert len(text.splitlines()) == 3


def test_read_skips_blank_lines_and_validates(tmp_path: Path) -> None:
    pred = Prediction(example_id="ex1", db_id="toy", predicted_sql="SELECT 1", raw_output="x")
    path = tmp_path / "p.jsonl"
    path.write_text("\n" + pred.model_dump_json() + "\n\n   \n", encoding="utf-8")
    assert read_jsonl(path, Prediction) == [pred]

    path.write_text('{"example_id": "ex1"}\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        read_jsonl(path, Prediction)


def test_write_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    write_jsonl(path, [])
    assert read_jsonl(path, Example) == []
