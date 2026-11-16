"""Remove duplicate SQL per database."""

from __future__ import annotations

from sqldistill.records import Example


def normalize_sql(sql: str) -> str:
    """Normalise SQL for duplicate detection.

    Strips, drops trailing semicolons, collapses whitespace runs to one space
    and lowercases everything outside single-quoted string literals. Literal
    contents (including their case and spacing) are kept: ``'Ana'`` and
    ``'ana'`` are different queries.
    """
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()

    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "'":
            # Copy the literal verbatim; '' inside it is an escaped quote.
            j = i + 1
            while j < n:
                if s[j] == "'":
                    if j + 1 < n and s[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(s[i : j + 1])  # an unterminated literal runs to the end
            i = j + 1
        elif ch.isspace():
            while i < n and s[i].isspace():
                i += 1
            out.append(" ")
        else:
            out.append(ch.lower())
            i += 1
    return "".join(out)


def dedup(examples: list[Example]) -> list[Example]:
    """Keep the first example of each (db_id, normalized sql), preserving order."""
    seen: set[tuple[str, str]] = set()
    kept: list[Example] = []
    for example in examples:
        key = (example.db_id, normalize_sql(example.sql))
        if key in seen:
            continue
        seen.add(key)
        kept.append(example)
    return kept
