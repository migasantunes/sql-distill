# Architecture

This document is the contract for the codebase. Module boundaries and the
signatures below are fixed; change them here first, then in code.

## Pipeline

```
 schemas (Spider train DBs)
        │
        ▼
 ┌──────────────┐  candidates.jsonl  ┌──────────────┐  train.jsonl  ┌────────────┐
 │ datagen      │ ─────────────────▶ │ filter       │ ────────────▶ │ train      │
 │ teacher LLM  │                    │ run on SQLite│               │ QLoRA      │
 └──────────────┘                    └──────────────┘               └─────┬──────┘
                                                                          │ adapter
 benchmark (Spider dev, BIRD Mini-Dev)                                    ▼
        │                                                           ┌────────────┐
        └─────────────────────────────────────────────────────────▶ │ eval       │
                                                                    │ exec. acc. │
                                                                    └─────┬──────┘
                                                                          ▼
                                                              results/*.json → report
```

Every stage reads files and writes files. No stage imports another stage's
internals; they share only `records`, `sqlexec`, `schema`, `prompts`, `config`.

## Layout

```
configs/                 YAML, one per stage, loaded into pydantic models
src/sqldistill/
  config.py              pydantic settings + YAML loader
  records.py             JSONL record types shared by every stage
  sqlexec.py             read-only SQLite execution + result comparison
  schema.py              SQLite introspection → schema text for prompts
  prompts.py             all prompt templates (teacher and student)
  splits.py              leakage guard: train/eval database disjointness
  teacher/               Teacher protocol, FakeTeacher, API-backed teacher (M2)
  datagen/               candidate generation, execution filter, dedup
  train/                 chat formatting (pure) + QLoRA loop (M4, GPU)
  infer/                 SQLGenerator protocol, HF backend (M3), SQL extraction
  eval/                  metrics, runner, benchmark loaders
  cli.py                 `sqldistill` entry point (typer)
tests/                   pytest; no network, no GPU, no downloads
data/                    gitignored: benchmarks, generated data
results/                 committed: small JSON result files + figures
```

## Dependency rule

Core install (`pip install -e .`) must stay light: pydantic, pyyaml, typer.
Heavy dependencies live in extras and are imported lazily inside functions:

| extra     | packages                                              | used by            |
|-----------|-------------------------------------------------------|--------------------|
| `train`   | torch, transformers, peft, bitsandbytes, datasets, accelerate | `train/lora.py`, `infer/hf.py` |
| `teacher` | anthropic                                             | `teacher/anthropic_teacher.py` |
| `demo`    | gradio                                                | `demo/` (M7)       |
| `dev`     | pytest, ruff                                          | tests, lint        |

`import sqldistill` and the whole test suite must work with only core + dev.

## Contracts

### `records.py`

```python
class Example(BaseModel):
    """One question/SQL pair. The unit every stage exchanges."""
    id: str                      # stable, unique within a file
    db_id: str                   # database name; the unit of train/eval splitting
    question: str
    sql: str
    source: Literal["teacher", "gold"]
    evidence: str | None = None  # BIRD external-knowledge hint, if any
    meta: dict[str, Any] = {}    # teacher model, temperature, filter verdicts…

class Prediction(BaseModel):
    example_id: str
    db_id: str
    predicted_sql: str
    raw_output: str              # untouched model text, before SQL extraction
    latency_s: float | None = None

def read_jsonl(path: Path, model: type[T]) -> list[T]
def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None   # creates parent dirs
```

### `sqlexec.py`

```python
class ExecStatus(str, Enum): OK = "ok"; ERROR = "error"; TIMEOUT = "timeout"   # not StrEnum: Python 3.10

@dataclass(frozen=True)
class ExecResult:
    status: ExecStatus
    rows: list[tuple] | None     # None unless OK
    error: str | None            # None when OK
    elapsed_s: float

def connect_readonly(db_path: Path) -> sqlite3.Connection   # the only place a database is opened
def execute(db_path: Path, sql: str, *, timeout_s: float = 10.0,
            max_rows: int = 100_000) -> ExecResult
def results_match(a: list[tuple], b: list[tuple], *, ordered: bool = False) -> bool
def has_order_by(sql: str) -> bool
```

Rules:
- Databases are opened **read-only** (`file:<path>?mode=ro`, `uri=True`). A model
  that emits `DROP TABLE` gets `ERROR`, and the file is untouched.
- Timeout is enforced with `sqlite3.Connection.set_progress_handler`; no threads.
- `execute` never raises for bad SQL; it returns `ERROR`. It raises only for a
  missing database file (`FileNotFoundError`), which is a setup bug.
- Text is decoded with `errors="replace"`: some benchmark databases contain
  invalid UTF-8, and a decode error must not turn a gold query into `ERROR`.
- `max_rows` truncates silently and still returns `OK`. Two queries that agree
  on the first `max_rows` rows would compare equal; M1 must confirm no gold
  result in either benchmark reaches the cap (raise it if one does).
- `results_match` compares rows as multisets when `ordered=False`, as lists when
  `True`. Floats are compared after rounding to 6 decimals; `1` and `1.0` match.

### `schema.py`

```python
@dataclass(frozen=True)
class Column: name: str; type: str; primary_key: bool
@dataclass(frozen=True)
class ForeignKey: column: str; ref_table: str; ref_column: str
@dataclass(frozen=True)
class Table: name: str; columns: tuple[Column, ...]; foreign_keys: tuple[ForeignKey, ...]
@dataclass(frozen=True)
class Schema: db_id: str; tables: tuple[Table, ...]

def introspect(db_path: Path, db_id: str | None = None) -> Schema   # db_id defaults to file stem
def render_schema(schema: Schema, *, sample_rows: dict[str, list[tuple]] | None = None) -> str
def sample_rows(db_path: Path, schema: Schema, n: int = 3) -> dict[str, list[tuple]]
```

`render_schema` emits `CREATE TABLE` statements (columns, PKs, FKs), optionally
followed by a comment block of sample rows per table. Output is deterministic:
the same database always renders to the same string (it is part of the prompt,
so it is part of the experiment).

### `prompts.py`

```python
STUDENT_SYSTEM: str
def student_messages(schema_text: str, question: str, evidence: str | None = None) -> list[dict[str, str]]
def teacher_generation_prompt(schema_text: str, n: int, difficulty: str) -> str
def parse_teacher_response(text: str) -> list[tuple[str, str]]   # (question, sql); tolerant, never raises
```

The student prompt is used **identically** for training targets, the zero-shot
baseline and evaluation. One function, one format.

### `splits.py`

```python
class LeakageError(ValueError): ...
def assert_disjoint(train_db_ids: Iterable[str], eval_db_ids: Iterable[str]) -> None
def db_ids_of(examples: Iterable[Example]) -> set[str]
```

Called by `train` before training and by `eval` when it is given the training
file. A shared `db_id` is a hard failure, not a warning.

### `teacher/`

```python
class Teacher(Protocol):
    name: str
    def generate(self, schema_text: str, *, n: int, difficulty: str) -> list[tuple[str, str]]: ...

class FakeTeacher:            # teacher/fake.py — deterministic, for tests and dry runs
    def __init__(self, pairs: list[tuple[str, str]]): ...
class AnthropicTeacher:       # teacher/anthropic_teacher.py — M2; lazy `import anthropic`
```

### `datagen/`

```python
# generate.py
def generate_candidates(db_paths: dict[str, Path], teacher: Teacher, *,
                        per_db: int, difficulties: Sequence[str]) -> list[Example]

# filter.py
class FilterLevel(IntEnum):
    NONE = 0          # keep everything (ablation baseline)
    EXECUTES = 1      # SQL runs without error or timeout
    NON_EMPTY = 2     # ...and returns at least one row
    CONSISTENT = 3    # ...and agrees with a second, independent teacher answer (M2)

@dataclass(frozen=True)
class FilterReport: total: int; kept: int; dropped_by_reason: dict[str, int]

def filter_examples(examples: list[Example], db_paths: dict[str, Path],
                    level: FilterLevel, *, timeout_s: float = 10.0
                    ) -> tuple[list[Example], FilterReport]

# dedup.py
def normalize_sql(sql: str) -> str          # whitespace/case/trailing-semicolon normalisation
def dedup(examples: list[Example]) -> list[Example]   # by (db_id, normalized sql), first wins
```

Drop reasons reported: `error`, `timeout`, `empty`, `unknown_db` (the example's
database is not in `db_paths`). `total == kept + sum(dropped_by_reason.values())`.

`CONSISTENT` raises `NotImplementedError` in the scaffold (needs the second
teacher pass, M2). Levels 0–2 are implemented.

### `infer/`

```python
class SQLGenerator(Protocol):
    name: str
    def generate(self, messages: list[dict[str, str]]) -> str: ...

def extract_sql(raw: str) -> str      # infer/extract.py: strip code fences / prose, first statement
    # Heuristics (each must be re-checked on real model output at M3): drops <think> blocks,
    # prefers the first fenced block, cuts at the first top-level semicolon, and outside a
    # fence also cuts at the first blank line (to drop trailing explanations).
class HFGenerator:                    # infer/hf.py — M3; lazy torch/transformers imports
```

### `eval/`

```python
# benchmarks.py
@dataclass(frozen=True)
class Benchmark: name: str; examples: list[Example]; db_paths: dict[str, Path]
def load_spider(root: Path, split: str = "dev") -> Benchmark
def load_bird_minidev(root: Path) -> Benchmark

# metrics.py
@dataclass(frozen=True)
class Scored: example_id: str; db_id: str; correct: bool; pred_status: ExecStatus; error: str | None
def score_prediction(example: Example, prediction: Prediction, db_path: Path,
                     *, timeout_s: float = 10.0) -> Scored
def execution_accuracy(scored: Sequence[Scored]) -> float
def bootstrap_ci(scored: Sequence[Scored], *, n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]

# runner.py
def run_eval(benchmark: Benchmark, generator: SQLGenerator, *, out_dir: Path,
             limit: int | None = None, timeout_s: float = 10.0) -> dict[str, Any]
```

A prediction is correct iff gold and predicted SQL both execute and
`results_match(gold_rows, pred_rows, ordered=has_order_by(example.sql))`.
If the *gold* SQL fails to execute, `score_prediction` raises: that is a broken
benchmark setup, and silently scoring it as wrong would corrupt the metric.

`run_eval` writes `predictions.jsonl`, `scored.jsonl` and `summary.json`
(accuracy, CI, n, error/timeout/generator-failure counts, mean latency,
generator name, benchmark name) to
`out_dir` and returns the summary dict.

### `train/`

```python
# format.py — pure, tested
def to_chat_example(example: Example, schema_text: str) -> dict[str, list[dict[str, str]]]
    # {"messages": student_messages(...) + [{"role": "assistant", "content": example.sql}]}

# lora.py — M4; lazy imports; raises NotImplementedError in the scaffold
def train(cfg: TrainConfig) -> Path     # returns adapter directory
```

### `config.py`

```python
class ModelConfig(BaseModel): student: str; teacher: str
class DataGenConfig(BaseModel): per_db: int; difficulties: list[str]; filter_level: int; temperature: float
class TrainConfig(BaseModel): lora_r: int; lora_alpha: int; lora_dropout: float; lr: float;
                              epochs: int; batch_size: int; grad_accum: int; max_seq_len: int;
                              load_in_4bit: bool; seed: int
class EvalConfig(BaseModel): timeout_s: float; max_new_tokens: int; benchmarks: list[str]
class Config(BaseModel): model: ModelConfig; datagen: DataGenConfig; train: TrainConfig; eval: EvalConfig

def load_config(path: Path) -> Config     # unknown keys are an error (extra="forbid")
```

### `cli.py`

`sqldistill` with subcommands `schema`, `gen`, `filter`, `train`, `eval`.
`schema <db.sqlite>` and `filter` work in the scaffold. Commands whose backend
is a later milestone exit with code 2 and a message naming the milestone.

## Invariants (each has a test)

1. No eval database is ever seen in training (`splits.assert_disjoint`).
2. Execution never mutates a database (read-only open).
3. Student prompt format is identical across train, baseline and eval.
4. Schema rendering is deterministic.
5. A failing gold query is an error, never a silent "incorrect".
6. Core import works without torch, transformers or anthropic installed.
