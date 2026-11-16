"""Pull one SQL statement out of free-form model output."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
# First fenced block: ``` or ```sql (any language tag); an unclosed fence runs to the end.
_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)(?:```|\Z)", re.DOTALL)
_SQL_LABEL = re.compile(r"^\s*sql\s*:\s*", re.IGNORECASE)
_STARTS_SQL = re.compile(r"(?:SELECT|WITH)\b", re.IGNORECASE)
_LINE_START_SQL = re.compile(r"^[ \t]*(?:SELECT|WITH)\b", re.IGNORECASE | re.MULTILINE)
_TOKEN_SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)
# A bare "with" is common in prose, so mid-line WITH only counts as a CTE start.
_TOKEN_CTE = re.compile(r"\bWITH\s+(?:RECURSIVE\s+)?[\w\"`\[\]]+\s*(?:\([^)]*\)\s*)?AS\s*\(", re.I)

_QUOTES = {"'": "'", '"': '"', "`": "`", "[": "]"}


def _remove_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    # Some chat templates put the opening tag in the prompt, leaving only "</think>".
    parts = _THINK_CLOSE.split(text)
    return parts[-1]


def _first_statement(text: str, *, cut_at_blank_line: bool) -> str:
    """Text up to the first semicolon outside quotes (and, for unfenced text,
    up to the first blank line outside quotes, which usually starts prose)."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _QUOTES:
            close = _QUOTES[ch]
            j = i + 1
            while j < n:
                if text[j] == close:
                    if close != "]" and j + 1 < n and text[j + 1] == close:
                        j += 2  # doubled quote is an escape
                        continue
                    break
                j += 1
            i = j + 1
            continue
        if ch == ";":
            return text[:i]
        if cut_at_blank_line and ch == "\n":
            rest = text[i + 1 :]
            stripped = rest.lstrip(" \t\r")
            if stripped.startswith("\n"):
                return text[:i]
        i += 1
    return text


def _extract(raw: str) -> str:
    text = _remove_think(raw)
    fence = _FENCE.search(text)
    fenced = fence is not None
    if fence is not None:
        text = fence.group(1)
    text = _SQL_LABEL.sub("", text.strip(), count=1)
    if not _STARTS_SQL.match(text):
        match = (
            _LINE_START_SQL.search(text) or _TOKEN_SELECT.search(text) or _TOKEN_CTE.search(text)
        )
        if match is not None:
            start = match.start()
            # _LINE_START_SQL may match leading indentation; skip it.
            while start < len(text) and text[start] in " \t":
                start += 1
            text = text[start:]
    return _first_statement(text, cut_at_blank_line=not fenced).strip()


def extract_sql(raw: str) -> str:
    """Return the first SQL statement in ``raw``, or "" if there is nothing.

    Steps, in order: drop ``<think>...</think>`` blocks; take the first fenced
    code block if there is one; strip a leading ``SQL:`` label; if the text does
    not start with SELECT/WITH, skip ahead to the first line (or token) that
    does; cut at the first semicolon outside quotes. Never raises.
    """
    if not raw:
        return ""
    try:
        return _extract(raw)
    except Exception:  # extraction must never crash an eval run
        return raw.strip()
