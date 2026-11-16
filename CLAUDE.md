# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`sqldistill`: distil text-to-SQL from a large teacher LLM into a ~1.7B student
using execution-verified synthetic data. A CV project whose value is in honest,
reproducible numbers. Read `docs/PLAN.md` (what and in which order) and
`docs/ARCHITECTURE.md` (contracts) before changing code.

## Commands

```bash
make install     # pip install -e ".[dev]"
make check       # ruff check + ruff format --check + pytest  (must pass before every commit)
make format      # auto-fix lint and formatting
pytest tests/test_sqlexec.py -q     # a single file
```

## Rules

- **Follow the milestones in `docs/PLAN.md` in order.** Each has acceptance
  criteria; do not start the next until they are met. The scaffold is M0.
- **`docs/ARCHITECTURE.md` is the contract.** To change a signature, update the
  document in the same commit and say why in `docs/DECISIONS.md`.
- **Heavy imports are lazy.** `torch`, `transformers`, `peft`, `anthropic`,
  `gradio` are imported inside functions only. `import sqldistill` and the test
  suite must work with core + dev dependencies alone.
- **Tests use no network, GPU or downloaded data.** Use the `toy_db` fixture and
  `FakeTeacher`. Code that needs a GPU or an API gets a thin wrapper around
  pure, tested functions.
- **Never weaken the invariants** listed at the end of `ARCHITECTURE.md`
  (leakage guard, read-only execution, one prompt format, failing gold = error).
  If a test for one of them is in the way, the change is wrong.
- **No number without a file.** Anything reported in the README must be
  regenerable from `results/`. Do not write results by hand.
- **Money and secrets.** API keys come from the environment (`ANTHROPIC_API_KEY`),
  never from files in the repo. Teacher calls must go through the on-disk cache
  and cost log (M2). Ask before any run expected to cost more than a few euros.
- **Never commit** `data/`, model weights, `.env`.

## Working style

- Miguel must be able to explain every part in an interview. When implementing
  a milestone, explain the non-obvious choices (why loss is masked to the
  answer, why comparison is order-insensitive, what QLoRA freezes) as you go,
  and prefer plain, readable code over clever code.
- Small commits, one concern each, imperative subject lines.
- Stubs raise `NotImplementedError("M<n>: ...")` naming their milestone; grep
  for `NotImplementedError` to see what is left.
