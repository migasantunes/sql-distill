from __future__ import annotations

from pathlib import Path

import pytest

from sqldistill.eval.metrics import Scored, bootstrap_ci, execution_accuracy, score_prediction
from sqldistill.records import Example, Prediction
from sqldistill.sqlexec import ExecStatus


def _ex(sql: str, id: str = "e1") -> Example:
    return Example(id=id, db_id="toy", question="q", sql=sql, source="gold")


def _pred(sql: str, id: str = "e1") -> Prediction:
    return Prediction(example_id=id, db_id="toy", predicted_sql=sql, raw_output=sql)


def _scored(correct: bool, i: int = 0) -> Scored:
    return Scored(f"e{i}", "toy", correct, ExecStatus.OK, None)


def test_correct_different_sql_same_rows(toy_db: Path) -> None:
    gold = _ex("SELECT name FROM singer WHERE country = 'Portugal'")
    pred = _pred("SELECT name FROM singer WHERE age > 30")
    s = score_prediction(gold, pred, toy_db)
    assert s.correct and s.pred_status is ExecStatus.OK and s.error is None


def test_order_ignored_without_order_by(toy_db: Path) -> None:
    gold = _ex("SELECT name FROM singer")
    pred = _pred("SELECT name FROM singer ORDER BY name DESC")
    assert score_prediction(gold, pred, toy_db).correct


def test_order_matters_with_order_by(toy_db: Path) -> None:
    gold = _ex("SELECT name FROM singer ORDER BY age")
    assert not score_prediction(
        gold, _pred("SELECT name FROM singer ORDER BY age DESC"), toy_db
    ).correct
    assert score_prediction(gold, _pred("SELECT name FROM singer ORDER BY age ASC"), toy_db).correct


def test_wrong_rows(toy_db: Path) -> None:
    s = score_prediction(_ex("SELECT name FROM singer"), _pred("SELECT title FROM concert"), toy_db)
    assert not s.correct and s.pred_status is ExecStatus.OK


def test_prediction_error(toy_db: Path) -> None:
    s = score_prediction(_ex("SELECT 1"), _pred("SELEC nonsense"), toy_db)
    assert not s.correct and s.pred_status is ExecStatus.ERROR and s.error


def test_empty_prediction_is_error(toy_db: Path) -> None:
    s = score_prediction(_ex("SELECT 1"), _pred("  "), toy_db)
    assert not s.correct and s.pred_status is ExecStatus.ERROR


def test_failing_gold_raises(toy_db: Path) -> None:
    # Invariant 5: a broken gold query is an error, never a silent "incorrect".
    with pytest.raises(RuntimeError, match="bad-gold"):
        score_prediction(
            _ex("SELECT * FROM nope", "bad-gold"), _pred("SELECT 1", "bad-gold"), toy_db
        )


def test_mismatched_ids_raise(toy_db: Path) -> None:
    with pytest.raises(ValueError):
        score_prediction(_ex("SELECT 1", "a"), _pred("SELECT 1", "b"), toy_db)


def test_execution_accuracy() -> None:
    assert execution_accuracy([_scored(True), _scored(False), _scored(True), _scored(True)]) == 0.75


def test_execution_accuracy_empty_raises() -> None:
    with pytest.raises(ValueError):
        execution_accuracy([])


def test_bootstrap_ci_contains_accuracy_and_is_deterministic() -> None:
    scored = [_scored(i % 3 != 0, i) for i in range(60)]
    acc = execution_accuracy(scored)
    low, high = bootstrap_ci(scored, seed=1)
    assert 0.0 <= low <= acc <= high <= 1.0
    assert low < high
    assert bootstrap_ci(scored, seed=1) == (low, high)


def test_bootstrap_ci_degenerate() -> None:
    assert bootstrap_ci([_scored(True, i) for i in range(10)]) == (1.0, 1.0)
    assert bootstrap_ci([_scored(False, i) for i in range(10)]) == (0.0, 0.0)


def test_bootstrap_ci_narrows_with_more_data() -> None:
    small = [_scored(i % 2 == 0, i) for i in range(20)]
    large = [_scored(i % 2 == 0, i) for i in range(2000)]
    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert hi_l - lo_l < hi_s - lo_s


def test_bootstrap_ci_empty_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
