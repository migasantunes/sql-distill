"""SQLite introspection and deterministic schema text for prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqldistill.sqlexec import connect_readonly

_PLAIN_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_SAMPLE_STR = 50


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]


@dataclass(frozen=True)
class Schema:
    db_id: str
    tables: tuple[Table, ...]


def _quote_always(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _ident(name: str) -> str:
    """Quote an identifier only when it is not a plain identifier."""
    return name if _PLAIN_IDENT.fullmatch(name) else _quote_always(name)


def introspect(db_path: Path, db_id: str | None = None) -> Schema:
    """Read tables, columns, primary and foreign keys from a SQLite file."""
    db_path = Path(db_path)
    conn = connect_readonly(db_path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
            )
        ]
        infos = {
            n: conn.execute(f"PRAGMA table_info({_quote_always(n)})").fetchall() for n in names
        }
        tables: list[Table] = []
        for name in names:
            # table_info rows: (cid, name, type, notnull, dflt_value, pk)
            info = sorted(infos[name], key=lambda r: r[0])
            columns = tuple(Column(r[1], r[2] or "", bool(r[5])) for r in info)
            # foreign_key_list rows: (id, seq, table, from, to, on_update, on_delete, match)
            fk_rows = sorted(
                conn.execute(f"PRAGMA foreign_key_list({_quote_always(name)})").fetchall(),
                key=lambda r: (r[0], r[1]),
            )
            fks = []
            for r in fk_rows:
                ref_table, ref_col = r[2], r[4]
                if ref_col is None:
                    # Implicit reference to the parent's primary key.
                    ref_info = infos.get(ref_table)
                    if ref_info is None:
                        ref_info = conn.execute(
                            f"PRAGMA table_info({_quote_always(ref_table)})"
                        ).fetchall()
                    pks = sorted((row for row in ref_info if row[5]), key=lambda row: row[5])
                    ref_col = pks[r[1]][1] if r[1] < len(pks) else ""
                fks.append(ForeignKey(r[3], ref_table, ref_col))
            tables.append(Table(name, columns, tuple(fks)))
    finally:
        conn.close()
    return Schema(db_id=db_id if db_id is not None else db_path.stem, tables=tuple(tables))


def _fmt_value(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        if len(v) > _MAX_SAMPLE_STR:
            v = v[:_MAX_SAMPLE_STR] + "..."
        return repr(v)
    if isinstance(v, (bytes, bytearray)):
        return f"<blob {len(v)} bytes>"
    return repr(v)


def _render_table(table: Table) -> str:
    pk_cols = [c.name for c in table.columns if c.primary_key]
    inline_pk = len(pk_cols) == 1
    lines: list[str] = []
    for col in table.columns:
        parts = [_ident(col.name)]
        if col.type:
            parts.append(col.type)
        if inline_pk and col.primary_key:
            parts.append("PRIMARY KEY")
        lines.append(" ".join(parts))
    if len(pk_cols) > 1:
        lines.append(f"PRIMARY KEY ({', '.join(_ident(c) for c in pk_cols)})")
    for fk in table.foreign_keys:
        ref = f"REFERENCES {_ident(fk.ref_table)}"
        if fk.ref_column:
            ref += f" ({_ident(fk.ref_column)})"
        lines.append(f"FOREIGN KEY ({_ident(fk.column)}) {ref}")
    body = ",\n".join(f"  {line}" for line in lines)
    return f"CREATE TABLE {_ident(table.name)} (\n{body}\n);"


def render_schema(schema: Schema, *, sample_rows: dict[str, list[tuple]] | None = None) -> str:
    """Render CREATE TABLE statements, optionally followed by sample-row comments."""
    blocks: list[str] = []
    for table in schema.tables:
        block = _render_table(table)
        if sample_rows is not None and table.name in sample_rows:
            rows = sample_rows[table.name]
            lines = ["-- sample rows:"]
            lines += [f"-- ({', '.join(_fmt_value(v) for v in row)})" for row in rows]
            if not rows:
                lines.append("-- (no rows)")
            block += "\n" + "\n".join(lines)
        blocks.append(block)
    return "\n\n".join(blocks) + "\n" if blocks else ""


def sample_rows(db_path: Path, schema: Schema, n: int = 3) -> dict[str, list[tuple]]:
    """First ``n`` rows of every table in ``schema``, read-only."""
    conn = connect_readonly(db_path)
    try:
        return {
            t.name: [
                tuple(r)
                for r in conn.execute(f"SELECT * FROM {_quote_always(t.name)} LIMIT ?", (n,))
            ]
            for t in schema.tables
        }
    finally:
        conn.close()
