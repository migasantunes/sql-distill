"""``sqldistill`` command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from sqldistill.config import Config, load_config
from sqldistill.records import Example, read_jsonl, write_jsonl
from sqldistill.schema import introspect, render_schema, sample_rows

app = typer.Typer(
    help="Distil text-to-SQL from a teacher LLM into a small model, "
    "using execution-verified synthetic data.",
    no_args_is_help=True,
    add_completion=False,
)

_DEFAULT_CONFIG = Path("configs/base.yaml")
_SAMPLE_ROWS = 3

ConfigOption = Annotated[
    Path, typer.Option("--config", help="Experiment config (YAML).", dir_okay=False)
]


def _load(config: Path) -> Config:
    try:
        return load_config(config)
    except FileNotFoundError:
        typer.echo(f"config not found: {config}", err=True)
        raise typer.Exit(1) from None
    except ValidationError as exc:
        typer.echo(f"invalid config {config}:\n{exc}", err=True)
        raise typer.Exit(1) from None


def _not_yet(command: str, milestone: str) -> None:
    typer.echo(
        f"`sqldistill {command}` is implemented in {milestone} (see docs/PLAN.md).", err=True
    )
    raise typer.Exit(2)


def _find_db(db_dir: Path, db_id: str) -> Path | None:
    for candidate in (db_dir / db_id / f"{db_id}.sqlite", db_dir / f"{db_id}.sqlite"):
        if candidate.is_file():
            return candidate
    return None


@app.command()
def schema(
    db_path: Annotated[
        Path, typer.Argument(help="SQLite database file.", exists=True, dir_okay=False)
    ],
    samples: Annotated[
        bool, typer.Option("--samples/--no-samples", help="Append 3 sample rows per table.")
    ] = True,
) -> None:
    """Print the schema text the prompts use for DB_PATH."""
    s = introspect(db_path)
    rows = sample_rows(db_path, s, n=_SAMPLE_ROWS) if samples else None
    typer.echo(render_schema(s, sample_rows=rows), nl=False)


@app.command("filter")
def filter_cmd(
    in_jsonl: Annotated[Path, typer.Argument(help="Input Examples (JSONL).", exists=True)],
    out_jsonl: Annotated[Path, typer.Argument(help="Where to write the kept Examples.")],
    db_dir: Annotated[
        Path,
        typer.Option(
            "--db-dir",
            help="Directory with <db_id>/<db_id>.sqlite or <db_id>.sqlite.",
            exists=True,
            file_okay=False,
        ),
    ],
    level: Annotated[
        int, typer.Option("--level", help="0 none, 1 executes, 2 non-empty, 3 consistent.")
    ],
) -> None:
    """Run the execution filter over IN_JSONL and write the kept examples."""
    from sqldistill.datagen.filter import FilterLevel, filter_examples

    try:
        filter_level = FilterLevel(level)
    except ValueError:
        raise typer.BadParameter(f"must be one of 0-3, got {level}", param_hint="--level") from None

    examples = read_jsonl(in_jsonl, Example)
    db_paths: dict[str, Path] = {}
    missing: list[str] = []
    for db_id in sorted({ex.db_id for ex in examples}):
        path = _find_db(db_dir, db_id)
        if path is None:
            missing.append(db_id)
        else:
            db_paths[db_id] = path
    if missing:
        typer.echo(f"warning: no database in {db_dir} for: {', '.join(missing)}", err=True)

    try:
        kept, report = filter_examples(examples, db_paths, filter_level)
    except NotImplementedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    write_jsonl(out_jsonl, kept)

    typer.echo(
        f"level {int(filter_level)} ({filter_level.name}): kept {report.kept}/{report.total}"
    )
    for reason, count in report.dropped_by_reason.items():
        typer.echo(f"  dropped {reason}: {count}")


@app.command()
def gen(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Generate synthetic candidates with the teacher."""
    _load(config)
    _not_yet("gen", "M2")


@app.command()
def train(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Fine-tune the student with QLoRA."""
    _load(config)
    _not_yet("train", "M4")


@app.command("eval")
def eval_cmd(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Evaluate a model on the configured benchmarks."""
    _load(config)
    _not_yet("eval", "M3")


if __name__ == "__main__":
    app()
