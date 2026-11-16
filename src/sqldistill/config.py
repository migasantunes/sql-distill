"""Experiment configuration: pydantic models and YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student: str
    teacher: str


class DataGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_db: int = Field(ge=1)
    difficulties: list[str] = Field(min_length=1)
    filter_level: int = Field(ge=0, le=3)
    temperature: float = Field(ge=0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    lr: float = Field(gt=0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    grad_accum: int = Field(ge=1)
    max_seq_len: int = Field(gt=0)
    load_in_4bit: bool
    seed: int


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    benchmarks: list[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    datagen: DataGenConfig
    train: TrainConfig
    eval: EvalConfig


def load_config(path: Path) -> Config:
    """Load and validate a YAML config. Unknown keys raise ValidationError."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config.model_validate(data)
