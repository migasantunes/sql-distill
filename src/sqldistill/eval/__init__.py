"""Evaluation: benchmark loaders, execution-accuracy metrics and the eval runner."""

from __future__ import annotations

from sqldistill.eval.benchmarks import Benchmark, load_bird_minidev, load_spider
from sqldistill.eval.metrics import Scored, bootstrap_ci, execution_accuracy, score_prediction
from sqldistill.eval.runner import run_eval

__all__ = [
    "Benchmark",
    "Scored",
    "bootstrap_ci",
    "execution_accuracy",
    "load_bird_minidev",
    "load_spider",
    "run_eval",
    "score_prediction",
]
