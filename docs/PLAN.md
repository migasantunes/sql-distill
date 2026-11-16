# Work plan

## Goal

Show, with numbers that survive questioning, how close a ~1.7B-parameter model
can get to a frontier model on text-to-SQL when it is fine-tuned only on
synthetic data that the frontier model wrote and a database verified.

Headline claim to earn (fill in at M6):

> A 1.7B model fine-tuned on N execution-verified synthetic examples reaches X%
> execution accuracy on Spider dev (teacher: Y%, same model zero-shot: Z%),
> at roughly 1/K of the teacher's cost per query.

## Fixed choices

| Choice | Value | Why |
|---|---|---|
| Student | `Qwen/Qwen3-1.7B` (Apache 2.0) | Fits a free 16 GB T4 with 4-bit QLoRA; permissive licence |
| Second student (stretch) | `meta-llama/Llama-3.2-1B-Instruct` | Reported as highly tunable; tests whether results generalise across families |
| Teacher | Config-driven; default a current Claude Sonnet model via API | Strong SQL, cheap enough for ~10k pairs |
| Training schemas | Spider **train-split** databases (schemas + contents only) | Real, varied SQLite databases; disjoint from all eval databases |
| Primary benchmark | Spider dev (unseen databases) | Small models score in a meaningful range; standard execution metric |
| Hard benchmark | BIRD Mini-Dev (SQLite) | Dirtier data, external knowledge; shows where the student breaks |
| Metric | Execution accuracy + 95% bootstrap CI | A query returns the right rows or it does not |
| Compute | Free Colab/Kaggle T4 | Whole project reproducible at ~zero GPU cost |

Verify at M0 that both model IDs and dataset download locations still resolve;
swap in `configs/base.yaml` if not. Nothing in code hardcodes them.

Known caveat to state in the README: BIRD has documented annotation noise, so
BIRD numbers are reported as a stress test, not as the headline.

## Experiments

Systems compared on both benchmarks (same prompt, same decoding):

1. Student zero-shot (no fine-tuning): the floor.
2. Student + QLoRA on synthetic data: **the project**.
3. Student + QLoRA on Spider's human-written train set: reference for "what
   real labels buy you".
4. Teacher zero-shot: the ceiling.

Ablations (Spider dev, 3 seeds each where affordable):

| Ablation | Values | Question answered |
|---|---|---|
| Execution filter | level 0 / 1 / 2 / 3 | Does verifying the teacher's SQL matter? |
| Dataset size | 500 / 2k / 5k / 10k | Where does more synthetic data stop helping? |
| LoRA rank | 8 / 16 / 64 | Is capacity the bottleneck? |
| Sample rows in prompt | on / off | Does seeing data values help the student? |

Budget guard: each fine-tuning run should be under ~90 min on a T4. If the full
grid does not fit, cut seeds before cutting ablations, and say so in the report.

## Milestones

Each milestone ends with something runnable and a short entry in
`docs/DECISIONS.md` for anything that deviated from this plan.

### M0 — Scaffold (done in this commit)
Package skeleton, contracts in `ARCHITECTURE.md`, execution engine, schema
rendering, execution filter (levels 0–2), metrics, leakage guard, CLI shell,
tests, CI.
**Accept:** `make check` green on a clean clone with only core + dev deps.

### M1 — Data in
`scripts/` or `sqldistill data` command to download Spider and BIRD Mini-Dev
into `data/`. Implement `load_spider` / `load_bird_minidev` against the real
files. Sanity script: run every gold query, report failures.
**Accept:** gold queries execute on ≥99% of Spider dev; failures listed and
explained; `assert_disjoint(spider_train_dbs, spider_dev_dbs ∪ bird_dbs)` passes.

### M2 — Teacher and synthetic data
`AnthropicTeacher` with retries, rate limiting, on-disk response cache (so a
crash never re-spends money) and a token/cost log. Generate candidates across
difficulty tiers. Implement filter level 3 (second independent teacher answer
must return the same rows).
**Accept:** ≥10k candidates on disk; `FilterReport` per level committed to
`results/`; total API spend recorded. Hand-inspect 50 random kept pairs and
record the error rate you find.

### M3 — Inference and baselines
`HFGenerator` (4-bit load, greedy decoding, batched). Run student zero-shot and
teacher zero-shot on both benchmarks.
**Accept:** systems 1 and 4 have `summary.json` with CIs; a 20-example spot
check of `extract_sql` on raw outputs shows no extraction-caused failures (look
in particular for queries cut short at a blank line); accuracy cross-checked
against the official Spider evaluation script on the same predictions.

### M4 — Fine-tuning
`train/lora.py` with PEFT + bitsandbytes, loss masked to the assistant turn
only, checkpointing to Drive/Kaggle output, a Colab notebook that is a thin
wrapper over the CLI. Calls `assert_disjoint` before training.
**Accept:** one end-to-end run (2k examples) beats zero-shot on Spider dev with
non-overlapping CIs; training loss curve saved.

### M5 — Ablations
Run the grid above. One results file per run, named by its config hash.
**Accept:** every cell has a number or a written reason why not.

### M6 — Report
`results/` → figures (accuracy vs. dataset size; filter level bar chart; error
breakdown) and a README that leads with the headline number, the comparison
table and the caveats. Cost per query: measured tokens/s for the student on a
stated machine vs. teacher API price at the date of measurement.
**Accept:** every number in the README is regenerable from a file in `results/`.

### M7 — Demo and release
Gradio demo (question → SQL → executed rows) on a Hugging Face Space; adapter
pushed to the Hub with a model card. Optional: run the adapter-merged model
through `inference-ladder`.
**Accept:** public link in the README works from a clean browser.

## Risks

| Risk | Mitigation |
|---|---|
| Teacher writes plausible but wrong SQL | Filter levels 2–3; hand audit in M2; ablation quantifies the damage |
| Student fails on long schemas (context) | `max_seq_len` guard; drop over-long training examples and count them; report eval truncations |
| Synthetic questions are easier than benchmark questions | Difficulty tiers in the generation prompt; compare SQL feature histograms (joins, nesting, aggregates) against Spider dev |
| Leakage through schemas | `assert_disjoint` is a hard failure; test covers it |
| Free GPU session dies mid-run | Checkpoint every N steps; resumable training |
| Result is underwhelming | Still a result: the ablations explain why. Report it straight |

## Interview notes to be able to defend

- Why execution accuracy and not exact match.
- Why the result comparison is order-insensitive unless the gold query has `ORDER BY`.
- What QLoRA trains and what it freezes; why loss is masked to the answer.
- Why the split is by database and what would inflate the score if it were not.
- What the filter ablation showed, including if it showed little.
