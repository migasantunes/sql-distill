from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

from sqldistill.sqlexec import ExecStatus, execute, has_order_by, results_match


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- execute ---------------------------------------------------------------


def test_select_ok(toy_db: Path) -> None:
    res = execute(toy_db, "SELECT name, age FROM singer WHERE country = 'Portugal' ORDER BY age")
    assert res.status is ExecStatus.OK
    assert res.rows == [("Ana", 31), ("Bruno", 45)]
    assert all(isinstance(r, tuple) for r in res.rows)
    assert res.error is None
    assert res.elapsed_s >= 0


def test_status_serializes_by_value() -> None:
    assert ExecStatus.TIMEOUT.value == "timeout"
    assert ExecStatus("ok") is ExecStatus.OK


@pytest.mark.parametrize(
    "sql",
    ["SELEC name FROM singer", "SELECT nope FROM singer", "SELECT * FROM missing", "", "   \n"],
)
def test_bad_sql_is_error_not_exception(toy_db: Path, sql: str) -> None:
    res = execute(toy_db, sql)
    assert res.status is ExecStatus.ERROR
    assert res.rows is None
    assert res.error


def test_multiple_statements_is_error(toy_db: Path) -> None:
    res = execute(toy_db, "SELECT 1; SELECT 2")
    assert res.status is ExecStatus.ERROR


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE singer",
        "DELETE FROM singer",
        "INSERT INTO singer VALUES (4, 'Dara', 'Spain', 22)",
        "UPDATE singer SET age = 0",
        "CREATE TABLE evil (x INTEGER)",
    ],
)
def test_writes_fail_and_db_is_untouched(toy_db: Path, sql: str) -> None:
    """Invariant 2: execution never mutates a database."""
    before = _digest(toy_db)
    res = execute(toy_db, sql)
    assert res.status is ExecStatus.ERROR
    assert _digest(toy_db) == before

    conn = sqlite3.connect(toy_db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        count = conn.execute("SELECT count(*) FROM singer").fetchone()[0]
    finally:
        conn.close()
    assert tables == {"singer", "concert"}
    assert count == 3


def test_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute(tmp_path / "nope.sqlite", "SELECT 1")
    assert not (tmp_path / "nope.sqlite").exists()


def test_path_with_special_characters(toy_db: Path, tmp_path: Path) -> None:
    odd = tmp_path / "dir with space?#&" / "my db#1?.sqlite"
    odd.parent.mkdir()
    shutil.copy(toy_db, odd)
    res = execute(odd, "SELECT count(*) FROM concert")
    assert res.status is ExecStatus.OK
    assert res.rows == [(3,)]


def test_timeout(toy_db: Path) -> None:
    forever = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    )
    res = execute(toy_db, forever, timeout_s=0.2)
    assert res.status is ExecStatus.TIMEOUT
    assert res.rows is None
    assert res.error
    assert res.elapsed_s < 5


def test_max_rows_cap(toy_db: Path) -> None:
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 1000) SELECT x FROM c"
    )
    res = execute(toy_db, sql, max_rows=10)
    assert res.status is ExecStatus.OK
    assert res.rows == [(i,) for i in range(1, 11)]


def test_non_ascii_text(toy_db: Path) -> None:
    res = execute(toy_db, "SELECT name FROM singer WHERE country = 'France'")
    assert res.rows == [("Chloé",)]


# --- results_match ----------------------------------------------------------


def test_unordered_is_multiset() -> None:
    assert results_match([(1,), (2,)], [(2,), (1,)])
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)])
    assert not results_match([(1,)], [(1,), (1,)])


def test_ordered_is_list() -> None:
    assert results_match([(1,), (2,)], [(1,), (2,)], ordered=True)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)


def test_numeric_normalisation() -> None:
    assert results_match([(1,)], [(1.0,)])
    assert results_match([(0.1 + 0.2,)], [(0.3,)])
    assert results_match([(1.0000001,)], [(1.0,)])
    assert not results_match([(1.001,)], [(1.0,)])


def test_none_and_mixed_types() -> None:
    a = [(None, "x", b"\x00"), (1, None, 2.5), ("a", 3, None)]
    b = [("a", 3.0, None), (None, "x", b"\x00"), (1.0, None, 2.5)]
    assert results_match(a, b)
    assert not results_match([(None,)], [(0,)])
    assert not results_match([("1",)], [(1,)])


def test_empty_results() -> None:
    assert results_match([], [])
    assert not results_match([], [(1,)])


def test_column_count_matters() -> None:
    assert not results_match([(1, 2)], [(1,)])


# --- has_order_by -----------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM singer ORDER BY age", True),
        ("select name from singer order\n  by age desc", True),
        ("SELECT name FROM singer", False),
        ("SELECT * FROM (SELECT name FROM singer ORDER BY age) LIMIT 1", False),
        ("SELECT name FROM singer WHERE name IN (SELECT name FROM s ORDER BY x)", False),
        ("SELECT * FROM singer WHERE name = 'order by'", False),
        ('SELECT "order by" FROM t', False),
        ("SELECT 'it''s order by' FROM t", False),
        ("SELECT a FROM t -- order by a\n", False),
        ("SELECT a FROM t /* order by a */", False),
        ("SELECT rank() OVER (ORDER BY a) FROM t", False),
        ("SELECT a FROM t UNION SELECT a FROM u ORDER BY a", True),
        ("SELECT * FROM (SELECT a FROM t) ORDER BY a", True),
        ("SELECT border, bye FROM t", False),
    ],
)
def test_has_order_by(sql: str, expected: bool) -> None:
    assert has_order_by(sql) is expected
