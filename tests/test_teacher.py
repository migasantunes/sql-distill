from __future__ import annotations

import importlib
import sys

import pytest

from sqldistill.teacher import FakeTeacher, Teacher

PAIRS = [("Q1", "SELECT 1"), ("Q2", "SELECT 2"), ("Q3", "SELECT 3")]


def test_fake_teacher_satisfies_protocol() -> None:
    teacher = FakeTeacher(PAIRS)
    assert isinstance(teacher, Teacher)
    assert teacher.name == "fake"


def test_fake_teacher_returns_first_n_and_records_calls() -> None:
    teacher = FakeTeacher(PAIRS)
    assert teacher.generate("schema", n=2, difficulty="simple") == PAIRS[:2]
    assert teacher.generate("schema", n=2, difficulty="simple") == PAIRS[:2]  # deterministic
    assert teacher.calls == [
        {"schema_text": "schema", "n": 2, "difficulty": "simple"},
        {"schema_text": "schema", "n": 2, "difficulty": "simple"},
    ]


def test_fake_teacher_does_not_cycle() -> None:
    teacher = FakeTeacher(PAIRS)
    assert teacher.generate("s", n=10, difficulty="moderate") == PAIRS
    assert FakeTeacher([]).generate("s", n=3, difficulty="simple") == []


def test_fake_teacher_result_is_a_copy() -> None:
    teacher = FakeTeacher(PAIRS)
    teacher.generate("s", n=3, difficulty="simple").clear()
    assert teacher.generate("s", n=3, difficulty="simple") == PAIRS


def test_anthropic_teacher_is_a_lazy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing must not need the anthropic package (invariant 6).
    monkeypatch.setitem(sys.modules, "anthropic", None)
    sys.modules.pop("sqldistill.teacher.anthropic_teacher", None)
    module = importlib.import_module("sqldistill.teacher.anthropic_teacher")

    teacher = module.AnthropicTeacher("some-model", temperature=0.5, max_tokens=100)
    assert teacher.name == "some-model"
    assert (teacher.temperature, teacher.max_tokens) == (0.5, 100)
    assert isinstance(teacher, Teacher)
    with pytest.raises(NotImplementedError, match="M2"):
        teacher.generate("schema", n=1, difficulty="simple")


def test_anthropic_teacher_defaults() -> None:
    from sqldistill.teacher.anthropic_teacher import AnthropicTeacher

    teacher = AnthropicTeacher("m")
    assert (teacher.temperature, teacher.max_tokens) == (0.8, 4096)
