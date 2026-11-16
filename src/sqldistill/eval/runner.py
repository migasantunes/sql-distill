"""Run a generator over a benchmark and write predictions, scores and a summary."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from sqldistill import prompts
from sqldistill.eval.benchmarks import Benchmark
from sqldistill.eval.metrics import Scored, bootstrap_ci, execution_accuracy, score_prediction
from sqldistill.infer.base import SQLGenerator
from sqldistill.infer.extract import extract_sql
from sqldistill.records import Prediction, write_jsonl
from sqldistill.schema import introspect, render_schema, sample_rows
from sqldistill.sqlexec import ExecStatus

_SAMPLE_ROWS = 3


def _scored_to_json(s: Scored) -> str:
    row = dataclasses.asdict(s)
    row["pred_status"] = s.pred_status.value
    return json.dumps(row, ensure_ascii=False)


def run_eval(
    benchmark: Benchmark,
    generator: SQLGenerator,
    *,
    out_dir: Path,
    limit: int | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Evaluate ``generator`` on ``benchmark`` (optionally its first ``limit`` examples).

    Writes ``predictions.jsonl``, ``scored.jsonl`` and ``summary.json`` to
    ``out_dir`` and returns the summary. A generator exception on one example is
    recorded as an empty, incorrect prediction instead of aborting the run.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    examples = benchmark.examples if limit is None else benchmark.examples[:limit]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_cache: dict[str, str] = {}
    predictions: list[Prediction] = []
    scored: list[Scored] = []
    n_failures = 0

    for ex in examples:
        if ex.db_id not in benchmark.db_paths:
            raise KeyError(f"benchmark {benchmark.name!r} has no database for {ex.db_id!r}")
        db_path = benchmark.db_paths[ex.db_id]
        if ex.db_id not in schema_cache:
            schema = introspect(db_path, ex.db_id)
            schema_cache[ex.db_id] = render_schema(
                schema, sample_rows=sample_rows(db_path, schema, n=_SAMPLE_ROWS)
            )
        messages = prompts.student_messages(schema_cache[ex.db_id], ex.question, ex.evidence)

        failure: str | None = None
        start = time.perf_counter()
        try:
            raw = generator.generate(messages)
            latency: float | None = time.perf_counter() - start
        except Exception as exc:  # one bad example must not kill the run
            raw, latency = "", None
            failure = f"generator failed: {type(exc).__name__}: {exc}"
            n_failures += 1

        pred = Prediction(
            example_id=ex.id,
            db_id=ex.db_id,
            predicted_sql=extract_sql(raw),
            raw_output=raw,
            latency_s=latency,
        )
        result = score_prediction(ex, pred, db_path, timeout_s=timeout_s)
        if failure is not None:
            result = dataclasses.replace(result, error=failure)
        predictions.append(pred)
        scored.append(result)

    write_jsonl(out_dir / "predictions.jsonl", predictions)
    with (out_dir / "scored.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for s in scored:
            fh.write(_scored_to_json(s) + "\n")

    latencies = [p.latency_s for p in predictions if p.latency_s is not None]
    ci_low, ci_high = bootstrap_ci(scored)
    summary: dict[str, Any] = {
        "benchmark": benchmark.name,
        "generator": generator.name,
        "n": len(scored),
        "accuracy": execution_accuracy(scored),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_error": sum(s.pred_status is ExecStatus.ERROR for s in scored),
        "n_timeout": sum(s.pred_status is ExecStatus.TIMEOUT for s in scored),
        "n_generator_failures": n_failures,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return summary
