from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from sqldistill.schema import (
    Column,
    ForeignKey,
    Schema,
    Table,
    introspect,
    render_schema,
    sample_rows,
)

EXPECTED_TOY = """\
CREATE TABLE concert (
  concert_id INTEGER PRIMARY KEY,
  title TEXT,
  year INTEGER,
  singer_id INTEGER,
  FOREIGN KEY (singer_id) REFERENCES singer (singer_id)
);

CREATE TABLE singer (
  singer_id INTEGER PRIMARY KEY,
  name TEXT,
  country TEXT,
  age INTEGER
);
"""


def test_introspect_toy(toy_db: Path) -> None:
    schema = introspect(toy_db)
    assert schema.db_id == "toy"
    assert [t.name for t in schema.tables] == ["concert", "singer"]
    singer = schema.tables[1]
    assert singer.columns == (
        Column("singer_id", "INTEGER", True),
        Column("name", "TEXT", False),
        Column("country", "TEXT", False),
        Column("age", "INTEGER", False),
    )
    assert singer.foreign_keys == ()
    assert schema.tables[0].foreign_keys == (ForeignKey("singer_id", "singer", "singer_id"),)


def test_introspect_db_id_override(toy_db: Path) -> None:
    assert introspect(toy_db, db_id="concert_singer").db_id == "concert_singer"


def test_introspect_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        introspect(tmp_path / "missing.sqlite")


def test_render_toy(toy_db: Path) -> None:
    assert render_schema(introspect(toy_db)) == EXPECTED_TOY


def test_render_is_deterministic(toy_db: Path) -> None:
    """Invariant 4: the same database always renders to the same string."""
    s1 = introspect(toy_db)
    s2 = introspect(toy_db)
    rows = sample_rows(toy_db, s1)
    assert render_schema(s1) == render_schema(s2)
    assert render_schema(s1, sample_rows=rows) == render_schema(
        s2, sample_rows=sample_rows(toy_db, s2)
    )


def test_introspection_does_not_mutate(toy_db: Path) -> None:
    before = hashlib.sha256(toy_db.read_bytes()).hexdigest()
    schema = introspect(toy_db)
    sample_rows(toy_db, schema)
    assert hashlib.sha256(toy_db.read_bytes()).hexdigest() == before


def test_sample_rows(toy_db: Path) -> None:
    schema = introspect(toy_db)
    rows = sample_rows(toy_db, schema, n=2)
    assert rows["singer"] == [(1, "Ana", "Portugal", 31), (2, "Bruno", "Portugal", 45)]
    assert len(rows["concert"]) == 2
    text = render_schema(schema, sample_rows=rows)
    assert "-- sample rows:\n-- (1, 'Ana', 'Portugal', 31)\n-- (2, 'Bruno', 'Portugal', 45)" in text


def test_composite_pk_quoting_and_samples(tmp_path: Path) -> None:
    path = tmp_path / "odd.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE "order items" (
            "order id" INTEGER, line INTEGER, note, "select" TEXT,
            PRIMARY KEY ("order id", line)
        );
        CREATE TABLE parent (id INTEGER PRIMARY KEY);
        CREATE TABLE child (pid INTEGER REFERENCES parent);
        INSERT INTO "order items" VALUES (1, 1, NULL, 'x');
        INSERT INTO "order items" VALUES (1, 2, X'00ff', 'y');
        """
    )
    conn.execute('INSERT INTO "order items" VALUES (2, 1, ?, ?)', ("Chloé", "z" * 80))
    conn.commit()
    conn.close()

    schema = introspect(path)
    assert [t.name for t in schema.tables] == ["child", "order items", "parent"]
    # Implicit FK reference resolves to the parent's primary key.
    assert schema.tables[0].foreign_keys == (ForeignKey("pid", "parent", "id"),)

    text = render_schema(schema, sample_rows=sample_rows(path, schema))
    assert 'CREATE TABLE "order items" (' in text
    assert '  "order id" INTEGER,\n  line INTEGER,\n  note,\n  select TEXT,' in text
    assert '  PRIMARY KEY ("order id", line)\n);' in text
    assert "FOREIGN KEY (pid) REFERENCES parent (id)" in text
    assert "-- (1, 1, NULL, 'x')" in text
    assert "-- (1, 2, <blob 2 bytes>, 'y')" in text
    assert f"-- (2, 1, 'Chloé', '{'z' * 50}...')" in text


def test_render_from_constructed_schema() -> None:
    schema = Schema(
        db_id="x",
        tables=(Table("t", (Column("a", "", False),), ()),),
    )
    assert render_schema(schema, sample_rows={"t": []}) == (
        "CREATE TABLE t (\n  a\n);\n-- sample rows:\n-- (no rows)\n"
    )
    assert render_schema(Schema("empty", ())) == ""
