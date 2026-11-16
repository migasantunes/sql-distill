from __future__ import annotations

import pytest

from sqldistill.datagen.dedup import dedup, normalize_sql
from sqldistill.records import Example


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("  SELECT  *\n\tFROM Singer ;  ", "select * from singer"),
        ("SELECT 1;;", "select 1"),
        ("SELECT 1 ; ;", "select 1"),
        (
            "SELECT Name FROM singer WHERE country = 'Portugal  PT'",
            "select name from singer where country = 'Portugal  PT'",
        ),
        ("SELECT 'It''s OK' AS X", "select 'It''s OK' as x"),
        ("SELECT 'a;'", "select 'a;'"),
        ("SELECT 'Unterminated  X", "select 'Unterminated  X"),
        ("", ""),
    ],
)
def test_normalize_sql(sql: str, expected: str) -> None:
    assert normalize_sql(sql) == expected


def test_normalize_keeps_literal_case_distinct() -> None:
    assert normalize_sql("SELECT * FROM t WHERE a = 'Ana'") != normalize_sql(
        "select * from t where a = 'ana'"
    )


def _ex(id_: str, db_id: str, sql: str) -> Example:
    return Example(id=id_, db_id=db_id, question=f"q {id_}", sql=sql, source="teacher")


def test_dedup_first_wins_and_order_preserved() -> None:
    examples = [
        _ex("a", "toy", "SELECT name FROM singer"),
        _ex("b", "toy", "SELECT COUNT(*) FROM concert"),
        _ex("c", "toy", "select  NAME from SINGER;"),
        _ex("d", "other", "SELECT name FROM singer"),  # same SQL, different db: kept
        _ex("e", "toy", "SELECT name FROM singer WHERE name = 'Ana'"),
        _ex("f", "toy", "SELECT name FROM singer WHERE name = 'ANA'"),  # literal differs: kept
        _ex("g", "other", "SELECT name FROM singer"),
    ]
    assert [e.id for e in dedup(examples)] == ["a", "b", "d", "e", "f"]


def test_dedup_empty() -> None:
    assert dedup([]) == []
