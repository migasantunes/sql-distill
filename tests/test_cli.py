from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqldistill.cli import app
from sqldistill.records import Example, read_jsonl, write_jsonl
from sqldistill.schema import introspect, render_schema, sample_rows

runner = CliRunner()


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "schema" in result.output and "filter" in result.output


def test_schema_with_samples(toy_db: Path) -> None:
    result = runner.invoke(app, ["schema", str(toy_db)])
    assert result.exit_code == 0, result.output
    s = introspect(toy_db)
    assert result.stdout == render_schema(s, sample_rows=sample_rows(toy_db, s, n=3))
    assert "-- sample rows:" in result.stdout


def test_schema_no_samples(toy_db: Path) -> None:
    result = runner.invoke(app, ["schema", str(toy_db), "--no-samples"])
    assert result.exit_code == 0, result.output
    assert result.stdout == render_schema(introspect(toy_db))


def _examples() -> list[Example]:
    return [
        Example(id="a", db_id="toy", question="q", sql="SELECT name FROM singer", source="teacher"),
        Example(id="b", db_id="toy", question="q", sql="SELEC broken", source="teacher"),
        Example(
            id="c",
            db_id="toy",
            question="q",
            sql="SELECT * FROM singer WHERE age > 99",
            source="teacher",
        ),
        Example(
            id="d", db_id="nested", question="q", sql="SELECT title FROM concert", source="teacher"
        ),
    ]


@pytest.mark.parametrize(
    ("level", "kept_ids"), [(0, ["a", "b", "c", "d"]), (1, ["a", "c", "d"]), (2, ["a", "d"])]
)
def test_filter_round_trip(toy_db: Path, tmp_path: Path, level: int, kept_ids: list[str]) -> None:
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    shutil.copy(toy_db, db_dir / "toy.sqlite")  # flat layout
    (db_dir / "nested").mkdir()
    shutil.copy(toy_db, db_dir / "nested" / "nested.sqlite")  # Spider layout
    src, dst = tmp_path / "in.jsonl", tmp_path / "out" / "kept.jsonl"
    write_jsonl(src, _examples())

    result = runner.invoke(
        app, ["filter", str(src), str(dst), "--db-dir", str(db_dir), "--level", str(level)]
    )
    assert result.exit_code == 0, result.output
    kept = read_jsonl(dst, Example)
    assert [ex.id for ex in kept] == kept_ids
    assert f"kept {len(kept_ids)}/4" in result.stdout


def test_filter_bad_level(toy_db: Path, tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    write_jsonl(src, _examples()[:1])
    args = ["filter", str(src), str(tmp_path / "o.jsonl"), "--db-dir", str(tmp_path), "--level"]
    assert runner.invoke(app, [*args, "7"]).exit_code != 0
    assert runner.invoke(app, [*args, "3"]).exit_code == 2  # level 3 is M2


@pytest.mark.parametrize(("command", "milestone"), [("gen", "M2"), ("train", "M4"), ("eval", "M3")])
def test_later_milestones_exit_2(command: str, milestone: str) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    result = runner.invoke(app, [command, "--config", str(config)])
    assert result.exit_code == 2
    assert milestone in result.stderr


def test_bad_config_exits_1(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {student: x, teacher: y, extra: 1}\n", encoding="utf-8")
    assert runner.invoke(app, ["train", "--config", str(bad)]).exit_code == 1
    assert runner.invoke(app, ["train", "--config", str(tmp_path / "none.yaml")]).exit_code == 1
