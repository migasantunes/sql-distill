from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sqldistill.eval.benchmarks import load_bird_minidev, load_spider
from sqldistill.sqlexec import ExecStatus, execute


def _place_db(toy_db: Path, db_root: Path, db_id: str) -> None:
    (db_root / db_id).mkdir(parents=True)
    shutil.copy(toy_db, db_root / db_id / f"{db_id}.sqlite")


@pytest.fixture
def spider_root(toy_db: Path, tmp_path: Path) -> Path:
    root = tmp_path / "spider"
    _place_db(toy_db, root / "database", "concert_singer")
    _place_db(toy_db, root / "database", "unused_db")
    items = [
        {
            "db_id": "concert_singer",
            "question": "How many singers?",
            "query": "SELECT count(*) FROM singer",
        },
        {
            "db_id": "concert_singer",
            "question": "Concert titles?",
            "query": "SELECT title FROM concert",
        },
    ]
    (root / "dev.json").write_text(json.dumps(items), encoding="utf-8")
    (root / "train_spider.json").write_text(json.dumps(items[:1]), encoding="utf-8")
    return root


@pytest.fixture
def bird_root(toy_db: Path, tmp_path: Path) -> Path:
    root = tmp_path / "bird"
    _place_db(toy_db, root / "dev_databases", "music")
    items = [
        {
            "question_id": 7,
            "db_id": "music",
            "question": "Singers from Portugal?",
            "evidence": "Portugal refers to country = 'Portugal'",
            "SQL": "SELECT name FROM singer WHERE country = 'Portugal'",
            "difficulty": "simple",
        },
        {"db_id": "music", "question": "Ages?", "evidence": "", "SQL": "SELECT age FROM singer"},
    ]
    (root / "mini_dev_sqlite.json").write_text(json.dumps(items), encoding="utf-8")
    return root


def test_load_spider_dev(spider_root: Path) -> None:
    bench = load_spider(spider_root)
    assert bench.name == "spider_dev"
    assert [ex.id for ex in bench.examples] == ["spider_dev-00000", "spider_dev-00001"]
    assert all(ex.source == "gold" for ex in bench.examples)
    assert bench.examples[0].sql == "SELECT count(*) FROM singer"
    assert set(bench.db_paths) == {"concert_singer"}  # only referenced dbs
    for ex in bench.examples:
        assert execute(bench.db_paths[ex.db_id], ex.sql).status is ExecStatus.OK


def test_load_spider_train(spider_root: Path) -> None:
    bench = load_spider(spider_root, split="train")
    assert bench.name == "spider_train" and len(bench.examples) == 1


def test_load_spider_unknown_split(spider_root: Path) -> None:
    with pytest.raises(ValueError):
        load_spider(spider_root, split="test")


def test_load_spider_missing_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="M1"):
        load_spider(tmp_path)


def test_load_spider_missing_db(spider_root: Path) -> None:
    shutil.rmtree(spider_root / "database" / "concert_singer")
    with pytest.raises(FileNotFoundError, match="concert_singer"):
        load_spider(spider_root)


def test_load_bird_minidev(bird_root: Path) -> None:
    bench = load_bird_minidev(bird_root)
    assert bench.name == "bird_minidev"
    first, second = bench.examples
    assert first.id == "bird_minidev-00000"
    assert first.evidence == "Portugal refers to country = 'Portugal'"
    assert first.meta == {"question_id": 7, "difficulty": "simple"}
    assert second.evidence is None
    assert bench.db_paths == {"music": bird_root / "dev_databases" / "music" / "music.sqlite"}


def test_load_bird_missing(tmp_path: Path, bird_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="docs/PLAN.md"):
        load_bird_minidev(tmp_path / "nowhere")
    shutil.rmtree(bird_root / "dev_databases")
    with pytest.raises(FileNotFoundError, match="music"):
        load_bird_minidev(bird_root)
