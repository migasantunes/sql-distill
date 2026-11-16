from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqldistill import prompts
from sqldistill.eval.benchmarks import Benchmark
from sqldistill.eval.runner import run_eval
from sqldistill.infer.base import SQLGenerator
from sqldistill.records import Example, Prediction, read_jsonl

QUESTIONS = {
    "How many singers are there?": "```sql\nSELECT COUNT(*) FROM singer;\n```",  # right
    "Names of Portuguese singers?": "SELECT name FROM singer",  # wrong rows
    "Oldest singer?": "I am not sure, sorry.",  # garbage
}


class CannedGenerator:
    name = "canned"

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        user = messages[-1]["content"]
        if self.fail_on is not None and self.fail_on in user:
            raise RuntimeError("CUDA out of memory")
        for question, answer in QUESTIONS.items():
            if question in user:
                return answer
        raise AssertionError("unexpected question")


def _bench(toy_db: Path) -> Benchmark:
    golds = [
        ("How many singers are there?", "SELECT COUNT(*) FROM singer"),
        ("Names of Portuguese singers?", "SELECT name FROM singer WHERE country = 'Portugal'"),
        ("Oldest singer?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),
    ]
    examples = [
        Example(id=f"toy-{i:05d}", db_id="toy", question=q, sql=s, source="gold")
        for i, (q, s) in enumerate(golds)
    ]
    return Benchmark(name="toybench", examples=examples, db_paths={"toy": toy_db})


def test_generator_protocol() -> None:
    assert isinstance(CannedGenerator(), SQLGenerator)


def test_run_eval_writes_files_and_summary(toy_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "run" / "a"
    gen = CannedGenerator()
    summary = run_eval(_bench(toy_db), gen, out_dir=out)

    assert summary["benchmark"] == "toybench"
    assert summary["generator"] == "canned"
    assert summary["n"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["ci_low"] <= summary["accuracy"] <= summary["ci_high"]
    assert summary["n_error"] == 1  # the garbage answer
    assert summary["n_timeout"] == 0
    assert summary["n_generator_failures"] == 0
    assert summary["mean_latency_s"] >= 0
    required = {"benchmark", "generator", "n", "accuracy", "ci_low", "ci_high"}
    assert required | {"n_error", "n_timeout", "mean_latency_s"} <= summary.keys()

    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    preds = read_jsonl(out / "predictions.jsonl", Prediction)
    assert [p.predicted_sql for p in preds] == [
        "SELECT COUNT(*) FROM singer",
        "SELECT name FROM singer",
        "I am not sure, sorry.",
    ]
    assert preds[0].raw_output == QUESTIONS["How many singers are there?"]
    scored = [json.loads(line) for line in (out / "scored.jsonl").read_text().splitlines()]
    assert [s["correct"] for s in scored] == [True, False, False]
    assert [s["pred_status"] for s in scored] == ["ok", "ok", "error"]


def test_run_eval_uses_student_prompt(toy_db: Path, tmp_path: Path) -> None:
    gen = CannedGenerator()
    bench = _bench(toy_db)
    run_eval(bench, gen, out_dir=tmp_path)
    first = gen.calls[0]
    assert first[0] == prompts.student_messages("x", "y")[0]  # same system prompt
    assert "CREATE TABLE singer" in first[-1]["content"]
    assert "-- sample rows:" in first[-1]["content"]


def test_run_eval_limit(toy_db: Path, tmp_path: Path) -> None:
    summary = run_eval(_bench(toy_db), CannedGenerator(), out_dir=tmp_path, limit=1)
    assert summary["n"] == 1 and summary["accuracy"] == 1.0


def test_generator_failure_does_not_kill_run(toy_db: Path, tmp_path: Path) -> None:
    gen = CannedGenerator(fail_on="How many singers")
    summary = run_eval(_bench(toy_db), gen, out_dir=tmp_path)
    assert summary["n"] == 3
    assert summary["n_generator_failures"] == 1
    assert summary["accuracy"] == 0.0
    preds = read_jsonl(tmp_path / "predictions.jsonl", Prediction)
    assert preds[0].raw_output == "" and preds[0].predicted_sql == ""
    first = json.loads((tmp_path / "scored.jsonl").read_text().splitlines()[0])
    assert first["correct"] is False and "generator failed" in first["error"]
