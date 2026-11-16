"""Read-only SQLite execution and result comparison."""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

# How many SQLite VM instructions between progress-handler calls.
_PROGRESS_STEPS = 1000


class ExecStatus(str, Enum):
    """Outcome of executing one query."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ExecResult:
    status: ExecStatus
    rows: list[tuple] | None  # None unless OK
    error: str | None  # None when OK
    elapsed_s: float


def _decode_text(raw: bytes) -> str:
    # Some benchmark databases contain invalid UTF-8; never fail on decoding.
    return raw.decode("utf-8", errors="replace")


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only. Raises FileNotFoundError if it does not exist."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = _decode_text
    return conn


def execute(
    db_path: Path,
    sql: str,
    *,
    timeout_s: float = 10.0,
    max_rows: int = 100_000,
) -> ExecResult:
    """Execute one SQL statement read-only. Never raises for bad SQL."""
    start = time.monotonic()
    conn = connect_readonly(db_path)  # FileNotFoundError propagates: setup bug
    try:
        if not sql or not sql.strip():
            return ExecResult(ExecStatus.ERROR, None, "empty SQL", time.monotonic() - start)

        deadline = start + timeout_s
        timed_out = False

        def _progress() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        conn.set_progress_handler(_progress, _PROGRESS_STEPS)
        try:
            cur = conn.execute(sql)
            rows = [tuple(r) for r in cur.fetchmany(max_rows)]
        except Exception as exc:  # sqlite3.Error, sqlite3.Warning, ValueError, ...
            elapsed = time.monotonic() - start
            if timed_out:
                return ExecResult(
                    ExecStatus.TIMEOUT, None, f"timed out after {timeout_s}s", elapsed
                )
            return ExecResult(ExecStatus.ERROR, None, f"{type(exc).__name__}: {exc}", elapsed)
        return ExecResult(ExecStatus.OK, rows, None, time.monotonic() - start)
    finally:
        conn.close()


def _norm_value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return v


def _norm_row(row: tuple) -> tuple:
    return tuple(_norm_value(v) for v in row)


def results_match(a: list[tuple], b: list[tuple], *, ordered: bool = False) -> bool:
    """Compare result sets: as lists if ``ordered``, else as multisets."""
    if len(a) != len(b):
        return False
    na = [_norm_row(r) for r in a]
    nb = [_norm_row(r) for r in b]
    if ordered:
        return na == nb
    return Counter(na) == Counter(nb)


_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _mask_nested(sql: str) -> str:
    """Return ``sql`` with literals, quoted identifiers, comments and anything
    inside parentheses replaced by spaces, so only the outermost query remains."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(sql)
    closers = {"'": "'", '"': '"', "`": "`", "[": "]"}
    while i < n:
        ch = sql[i]
        if ch in closers:
            close = closers[ch]
            j = i + 1
            while j < n:
                if sql[j] == close:
                    # Doubled quote is an escape inside quoted text.
                    if close != "]" and j + 1 < n and sql[j + 1] == close:
                        j += 2
                        continue
                    break
                j += 1
            out.append(" " * (min(j, n - 1) - i + 1))
            i = j + 1
            continue
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
            continue
        if ch == "(":
            depth += 1
            out.append(" ")
        elif ch == ")":
            depth = max(depth - 1, 0)
            out.append(" ")
        else:
            out.append(ch if depth == 0 else " ")
        i += 1
    return "".join(out)


def has_order_by(sql: str) -> bool:
    """True iff the outermost query has an ORDER BY clause."""
    return bool(_ORDER_BY_RE.search(_mask_nested(sql)))
