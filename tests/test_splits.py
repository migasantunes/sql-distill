from __future__ import annotations

import pytest

from sqldistill.records import Example
from sqldistill.splits import LeakageError, assert_disjoint, db_ids_of


def _ex(i: int, db_id: str) -> Example:
    return Example(id=str(i), db_id=db_id, question="q", sql="SELECT 1", source="teacher")


def test_disjoint_passes() -> None:
    assert_disjoint(["a", "b"], ["c", "d"])
    assert_disjoint([], ["c"])


def test_overlap_is_hard_failure_listing_sorted_ids() -> None:
    """Invariant 1: no eval database is ever seen in training."""
    with pytest.raises(LeakageError) as info:
        assert_disjoint(["zoo", "a", "concert_singer", "x"], ["concert_singer", "zoo", "q"])
    assert "concert_singer, zoo" in str(info.value)
    assert "x" not in str(info.value).split(":")[-1]
    assert isinstance(info.value, ValueError)


def test_accepts_generators() -> None:
    with pytest.raises(LeakageError):
        assert_disjoint((d for d in ["a"]), (d for d in ["a"]))


def test_db_ids_of_with_guard() -> None:
    train = [_ex(1, "a"), _ex(2, "a"), _ex(3, "b")]
    evals = [_ex(4, "c"), _ex(5, "b")]
    assert db_ids_of(train) == {"a", "b"}
    assert db_ids_of([]) == set()
    with pytest.raises(LeakageError, match="b"):
        assert_disjoint(db_ids_of(train), db_ids_of(evals))
