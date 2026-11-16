from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sqldistill.datagen.generate import generate_candidates, split_evenly
from sqldistill.schema import introspect, render_schema, sample_rows
from sqldistill.teacher import FakeTeacher

PAIRS = [
    ("How many singers are there?", "SELECT COUNT(*) FROM singer"),
    ("Which singers are from Portugal?", "SELECT name FROM singer WHERE country = 'Portugal'"),
    ("List concert titles.", "SELECT title FROM concert"),
]


def test_split_evenly() -> None:
    assert split_evenly(7, 3) == [3, 2, 2]
    assert split_evenly(6, 3) == [2, 2, 2]
    assert split_evenly(2, 3) == [1, 1, 0]


def test_generate_candidates_ids_meta_and_calls(toy_db: Path) -> None:
    teacher = FakeTeacher(PAIRS)
    examples = generate_candidates(
        {"toy": toy_db}, teacher, per_db=5, difficulties=["simple", "moderate"]
    )
    # 5 split over 2 tiers -> 3 + 2, remainder to the first tier.
    assert [c["n"] for c in teacher.calls] == [3, 2]
    assert [c["difficulty"] for c in teacher.calls] == ["simple", "moderate"]
    assert [e.id for e in examples] == [
        "toy-simple-0000",
        "toy-simple-0001",
        "toy-simple-0002",
        "toy-moderate-0000",
        "toy-moderate-0001",
    ]
    first = examples[0]
    assert (first.db_id, first.source, first.question, first.sql) == ("toy", "teacher", *PAIRS[0])
    assert first.meta == {"teacher": "fake", "difficulty": "simple"}
    assert examples[3].meta["difficulty"] == "moderate"


def test_generate_candidates_schema_has_sample_rows(toy_db: Path) -> None:
    teacher = FakeTeacher(PAIRS)
    generate_candidates({"toy": toy_db}, teacher, per_db=1, difficulties=["simple"])
    schema = introspect(toy_db, db_id="toy")
    expected = render_schema(schema, sample_rows=sample_rows(toy_db, schema, n=3))
    assert teacher.calls[0]["schema_text"] == expected
    assert "'Chloé'" in expected


def test_generate_candidates_skips_zero_tiers(toy_db: Path) -> None:
    teacher = FakeTeacher(PAIRS)
    generate_candidates(
        {"toy": toy_db}, teacher, per_db=2, difficulties=["simple", "moderate", "challenging"]
    )
    assert [(c["difficulty"], c["n"]) for c in teacher.calls] == [("simple", 1), ("moderate", 1)]


def test_generate_candidates_sorted_db_order(toy_db: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.sqlite"
    shutil.copy(toy_db, other)
    examples = generate_candidates(
        {"zeta": toy_db, "alpha": other},
        FakeTeacher(PAIRS),
        per_db=1,
        difficulties=["simple"],
    )
    assert [e.id for e in examples] == ["alpha-simple-0000", "zeta-simple-0000"]
    assert [e.db_id for e in examples] == ["alpha", "zeta"]


def test_generate_candidates_short_teacher(toy_db: Path) -> None:
    examples = generate_candidates(
        {"toy": toy_db}, FakeTeacher(PAIRS[:1]), per_db=4, difficulties=["simple"]
    )
    assert len(examples) == 1


@pytest.mark.parametrize(
    ("per_db", "difficulties"),
    [(0, ["simple"]), (3, []), (3, ["impossible"]), (3, ["simple", "simple"])],
)
def test_generate_candidates_bad_args(toy_db: Path, per_db: int, difficulties: list[str]) -> None:
    with pytest.raises(ValueError):
        generate_candidates(
            {"toy": toy_db}, FakeTeacher(PAIRS), per_db=per_db, difficulties=difficulties
        )
