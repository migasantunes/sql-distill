"""Execution accuracy and its bootstrap confidence interval."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqldistill.records import Example, Prediction
from sqldistill.sqlexec import ExecStatus, execute, has_order_by, results_match


@dataclass(frozen=True)
class Scored:
    example_id: str
    db_id: str
    correct: bool
    pred_status: ExecStatus
    error: str | None


def score_prediction(
    example: Example,
    prediction: Prediction,
    db_path: Path,
    *,
    timeout_s: float = 10.0,
) -> Scored:
    """Execute gold and predicted SQL and compare their results.

    Correct iff both execute and the rows match; row order matters only when
    the gold query has an outermost ORDER BY (otherwise the database is free
    to return rows in any order). A gold query that does not execute raises
    RuntimeError: that is a broken benchmark setup, not a wrong prediction.
    """
    if prediction.example_id != example.id:
        raise ValueError(
            f"prediction for {prediction.example_id!r} scored against example {example.id!r}"
        )
    gold = execute(db_path, example.sql, timeout_s=timeout_s)
    if gold.status is not ExecStatus.OK:
        raise RuntimeError(
            f"gold SQL for example {example.id!r} failed ({gold.status.value}): {gold.error}"
        )
    if not prediction.predicted_sql.strip():
        return Scored(example.id, example.db_id, False, ExecStatus.ERROR, "empty prediction")
    pred = execute(db_path, prediction.predicted_sql, timeout_s=timeout_s)
    if pred.status is not ExecStatus.OK:
        return Scored(example.id, example.db_id, False, pred.status, pred.error)
    assert gold.rows is not None and pred.rows is not None
    correct = results_match(gold.rows, pred.rows, ordered=has_order_by(example.sql))
    return Scored(example.id, example.db_id, correct, ExecStatus.OK, None)


def execution_accuracy(scored: Sequence[Scored]) -> float:
    """Fraction of predictions that are correct. Raises ValueError if empty."""
    if not scored:
        raise ValueError("execution_accuracy of an empty sequence is undefined")
    return sum(s.correct for s in scored) / len(scored)


def bootstrap_ci(
    scored: Sequence[Scored],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap (1 - alpha) confidence interval for execution accuracy.

    Resample the per-example correctness with replacement ``n_resamples`` times,
    compute the accuracy of each resample, and take the alpha/2 and 1 - alpha/2
    percentiles. Deterministic for a given ``seed``.
    """
    if not scored:
        raise ValueError("bootstrap_ci of an empty sequence is undefined")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = [1 if s.correct else 0 for s in scored]
    n = len(values)
    rng = random.Random(seed)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo_idx = max(0, math.floor(alpha / 2 * n_resamples))
    hi_idx = min(n_resamples - 1, math.ceil((1 - alpha / 2) * n_resamples) - 1)
    return stats[lo_idx], stats[hi_idx]
