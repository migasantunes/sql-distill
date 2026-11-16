from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sqldistill.config import Config, load_config

BASE_YAML = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def _base_dict() -> dict:
    return yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_real_base_config() -> None:
    cfg = load_config(BASE_YAML)
    assert isinstance(cfg, Config)
    assert cfg.model.student == "Qwen/Qwen3-1.7B"
    assert cfg.datagen.filter_level == 2
    assert cfg.datagen.difficulties == ["simple", "moderate", "challenging"]
    assert cfg.train.lr == pytest.approx(2.0e-4)
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.load_in_4bit is True
    assert cfg.eval.timeout_s == 10.0
    assert cfg.eval.benchmarks == ["spider_dev", "bird_minidev"]


@pytest.mark.parametrize("section", [None, "model", "datagen", "train", "eval"])
def test_unknown_key_is_error(tmp_path: Path, section: str | None) -> None:
    data = _base_dict()
    (data if section is None else data[section])["typo_key"] = 1
    with pytest.raises(ValidationError, match="typo_key"):
        load_config(_write(tmp_path, data))


def test_missing_key_is_error(tmp_path: Path) -> None:
    data = _base_dict()
    del data["train"]["lr"]
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("train", "lora_r", 0),
        ("train", "lr", 0.0),
        ("train", "lr", -1e-4),
        ("train", "epochs", 0),
        ("datagen", "filter_level", 4),
        ("datagen", "filter_level", -1),
        ("eval", "timeout_s", 0),
    ],
)
def test_invalid_values(tmp_path: Path, section: str, key: str, value: object) -> None:
    data = _base_dict()
    data[section][key] = value
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))
