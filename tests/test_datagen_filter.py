from __future__ import annotations

from pathlib import Path

import pytest

from sqldistill.datagen.filter import FilterLevel, FilterReport, filter_examples
from sqldistill.records import Example


def _ex(id_: str, sql: str, db_id: str = "toy") -> Example:
    return Example(id=id_, db_id=db_id, question="q", sql=sql, source="teacher", meta={"k": "v"})


@pytest.fixture
def examples() -> list[Example]:
    return [
        _ex("valid", "SELECT name FROM singer WHERE country = 'Portugal'"),
        _ex("syntax", "SELEC name FROM singer"),
        _ex("empty", "SELECT name FROM singer WHERE age > 100"),
        _ex("unknown", "SELECT 1", db_id="no_such_db"),
    ]


def _check_report(report: FilterReport) -> None:
    assert report.total == report.kept + sum(report.dropped_by_reason.values())


@pytest.mark.parametrize(
    ("level", "kept_ids", "dropped"),
    [
        (
            FilterLevel.NONE,
            ["valid", "syntax", "empty", "unknown"],
            {"error": 0, "timeout": 0, "empty": 0, "unknown_db": 0},
        ),
        (
            FilterLevel.EXECUTES,
            ["valid", "empty"],
            {"error": 1, "timeout": 0, "empty": 0, "unknown_db": 1},
        ),
        (
            FilterLevel.NON_EMPTY,
            ["valid"],
            {"error": 1, "timeout": 0, "empty": 1, "unknown_db": 1},
        ),
    ],
)
def test_filter_levels(
    toy_db: Path,
    examples: list[Example],
    level: FilterLevel,
    kept_ids: list[str],
    dropped: dict[str, int],
) -> None:
    kept, report = filter_examples(examples, {"toy": toy_db}, level)
    assert [e.id for e in kept] == kept_ids  # input order preserved
    assert report.total == 4
    assert report.kept == len(kept_ids)
    assert report.dropped_by_reason == dropped
    _check_report(report)
    for e in kept:
        assert e.meta == {"k": "v", "filter_level": int(level)}


def test_filter_accepts_plain_int(toy_db: Path, examples: list[Example]) -> None:
    kept, _ = filter_examples(examples, {"toy": toy_db}, 2)  # type: ignore[arg-type]
    assert [e.id for e in kept] == ["valid"]


def test_filter_does_not_mutate_inputs(toy_db: Path, examples: list[Example]) -> None:
    before = [e.model_copy(deep=True) for e in examples]
    kept, _ = filter_examples(examples, {"toy": toy_db}, FilterLevel.EXECUTES)
    assert examples == before
    assert all("filter_level" not in e.meta for e in examples)
    assert kept[0] is not examples[0]


def test_filter_timeout(toy_db: Path) -> None:
    slow = _ex(
        "slow",
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c",
    )
    kept, report = filter_examples([slow], {"toy": toy_db}, FilterLevel.EXECUTES, timeout_s=0.2)
    assert kept == []
    assert report.dropped_by_reason["timeout"] == 1
    _check_report(report)


def test_filter_never_writes(toy_db: Path) -> None:
    before = toy_db.read_bytes()
    kept, report = filter_examples(
        [_ex("drop", "DROP TABLE singer")], {"toy": toy_db}, FilterLevel.EXECUTES
    )
    assert kept == [] and report.dropped_by_reason["error"] == 1
    assert toy_db.read_bytes() == before


def test_filter_consistent_not_implemented(toy_db: Path, examples: list[Example]) -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        filter_examples(examples, {"toy": toy_db}, FilterLevel.CONSISTENT)


def test_filter_empty_input(toy_db: Path) -> None:
    kept, report = filter_examples([], {"toy": toy_db}, FilterLevel.NON_EMPTY)
    assert kept == [] and report.total == 0 and report.kept == 0
    _check_report(report)
