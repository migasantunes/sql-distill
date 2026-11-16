"""Execution filter: keep only teacher SQL that the database accepts.

Each level includes the checks of the levels below it. Level 0 exists so the
ablation can measure what filtering buys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from sqldistill.records import Example
from sqldistill.sqlexec import ExecStatus, execute

DROP_REASONS = ("error", "timeout", "empty", "unknown_db")


class FilterLevel(IntEnum):
    NONE = 0  # keep everything (ablation baseline)
    EXECUTES = 1  # SQL runs without error or timeout
    NON_EMPTY = 2  # ...and returns at least one row
    CONSISTENT = 3  # ...and agrees with a second, independent teacher answer (M2)


@dataclass(frozen=True)
class FilterReport:
    total: int
    kept: int
    dropped_by_reason: dict[str, int]


def _drop_reason(
    example: Example, db_paths: dict[str, Path], level: FilterLevel, timeout_s: float
) -> str | None:
    """Why ``example`` fails ``level``, or None if it passes."""
    path = db_paths.get(example.db_id)
    if path is None:
        return "unknown_db"
    result = execute(Path(path), example.sql, timeout_s=timeout_s)
    if result.status is ExecStatus.TIMEOUT:
        return "timeout"
    if result.status is not ExecStatus.OK:
        return "error"
    if level >= FilterLevel.NON_EMPTY and not result.rows:
        return "empty"
    return None


def filter_examples(
    examples: list[Example],
    db_paths: dict[str, Path],
    level: FilterLevel,
    *,
    timeout_s: float = 10.0,
) -> tuple[list[Example], FilterReport]:
    """Keep the examples that pass ``level``, in input order.

    Kept examples are copies whose ``meta`` gains ``filter_level``; inputs are
    not mutated. A missing database file raises FileNotFoundError (setup bug).
    """
    level = FilterLevel(level)
    if level is FilterLevel.CONSISTENT:
        raise NotImplementedError(
            "M2: filter level 3 needs a second, independent teacher answer per example"
        )

    dropped = dict.fromkeys(DROP_REASONS, 0)
    kept: list[Example] = []
    for example in examples:
        if level is not FilterLevel.NONE:
            reason = _drop_reason(example, db_paths, level, timeout_s)
            if reason is not None:
                dropped[reason] += 1
                continue
        kept.append(
            example.model_copy(update={"meta": {**example.meta, "filter_level": int(level)}})
        )
    report = FilterReport(total=len(examples), kept=len(kept), dropped_by_reason=dropped)
    return kept, report
