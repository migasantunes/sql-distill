from __future__ import annotations

import pytest

from sqldistill.infer.extract import extract_sql


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_empty_input(raw: str) -> None:
    assert extract_sql(raw) == ""


def test_plain_sql_unchanged() -> None:
    assert extract_sql("SELECT name FROM singer") == "SELECT name FROM singer"


def test_sql_fence() -> None:
    raw = "Here you go:\n```sql\nSELECT name FROM singer;\n```\nHope it helps."
    assert extract_sql(raw) == "SELECT name FROM singer"


def test_bare_fence() -> None:
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_first_fence_wins() -> None:
    raw = "```sql\nSELECT 1\n```\nor\n```sql\nSELECT 2\n```"
    assert extract_sql(raw) == "SELECT 1"


def test_unclosed_fence() -> None:
    assert extract_sql("```sql\nSELECT age FROM singer") == "SELECT age FROM singer"


def test_sql_label() -> None:
    assert extract_sql("SQL: SELECT 1") == "SELECT 1"
    assert extract_sql("sql:\nSELECT 1") == "SELECT 1"


def test_think_block_removed() -> None:
    raw = "<think>Maybe SELECT * FROM concert; no.</think>\nSELECT name FROM singer"
    assert extract_sql(raw) == "SELECT name FROM singer"


def test_think_block_with_fence_inside_removed() -> None:
    raw = "<think>\n```sql\nSELECT 0\n```\n</think>\n```sql\nSELECT 1\n```"
    assert extract_sql(raw) == "SELECT 1"


def test_dangling_think_close() -> None:
    assert extract_sql("reasoning here</think>\n\nSELECT 1") == "SELECT 1"


def test_prose_before_select_line() -> None:
    raw = "The query you need is:\nSELECT name\nFROM singer"
    assert extract_sql(raw) == "SELECT name\nFROM singer"


def test_prose_before_select_token() -> None:
    assert extract_sql("Answer: select count(*) from singer") == "select count(*) from singer"


def test_cte_start() -> None:
    raw = "Use a CTE:\nWITH t AS (SELECT 1 AS x) SELECT x FROM t"
    assert extract_sql(raw) == "WITH t AS (SELECT 1 AS x) SELECT x FROM t"


def test_prose_with_does_not_start_statement() -> None:
    raw = "Here it is with a join: SELECT s.name FROM singer s JOIN concert c USING (singer_id)"
    assert extract_sql(raw).startswith("SELECT s.name")


def test_cut_at_first_semicolon() -> None:
    assert extract_sql("SELECT 1; SELECT 2;") == "SELECT 1"


def test_semicolon_inside_quotes_kept() -> None:
    assert extract_sql("SELECT 'a;b' FROM t; DROP TABLE t") == "SELECT 'a;b' FROM t"
    assert extract_sql('SELECT "x;y" FROM t;') == 'SELECT "x;y" FROM t'
    assert extract_sql("SELECT 'it''s; fine' FROM t;") == "SELECT 'it''s; fine' FROM t"


def test_trailing_prose_after_blank_line_dropped() -> None:
    raw = "SELECT name FROM singer\n\nThis returns every singer's name."
    assert extract_sql(raw) == "SELECT name FROM singer"


def test_no_sql_returns_text() -> None:
    assert extract_sql("I don't know.") == "I don't know."


def test_whitespace_stripped() -> None:
    assert extract_sql("  \n SELECT 1 \n ") == "SELECT 1"
