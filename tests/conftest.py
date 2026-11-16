"""Shared fixtures. Tests never touch the network, a GPU or downloaded data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

TOY_DDL = """
CREATE TABLE singer (
    singer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT,
    age INTEGER
);
CREATE TABLE concert (
    concert_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    singer_id INTEGER,
    FOREIGN KEY (singer_id) REFERENCES singer (singer_id)
);
INSERT INTO singer VALUES
    (1, 'Ana', 'Portugal', 31),
    (2, 'Bruno', 'Portugal', 45),
    (3, 'Chloé', 'France', 27);
INSERT INTO concert VALUES
    (1, 'Spring Gala', 2023, 1),
    (2, 'Summer Fest', 2024, 1),
    (3, 'Winter Night', 2024, 3);
"""


@pytest.fixture
def toy_db(tmp_path: Path) -> Path:
    """A two-table SQLite database (singer, concert) with a foreign key."""
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(TOY_DDL)
        conn.commit()
    finally:
        conn.close()
    return path
