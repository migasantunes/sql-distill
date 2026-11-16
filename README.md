# sql-distill

Training a small model to write SQL by having a big one teach it.

A ~1.7B-parameter open model is fine-tuned to turn questions into SQL using
training data that a large teacher model writes and a real database verifies.
The question the project answers: **how much of a frontier model's text-to-SQL
accuracy can a laptop-sized model keep, and what does the verification step buy?**

> **Status: scaffold (M0).** The pipeline's core is implemented and tested;
> teacher generation, fine-tuning and results come in later milestones.
> See [docs/PLAN.md](docs/PLAN.md).

## Results

_To be filled at M6. Every number here will be regenerable from a file in `results/`._

| System | Spider dev (EX) | BIRD Mini-Dev (EX) |
|---|---|---|
| Student, zero-shot | – | – |
| Student + QLoRA, synthetic data | – | – |
| Student + QLoRA, Spider human labels | – | – |
| Teacher, zero-shot | – | – |

## How it works

1. **Generate.** A teacher LLM reads a database schema and writes question/SQL pairs.
2. **Verify.** Each query is run against the real SQLite database; pairs that
   fail, time out or return nothing are dropped.
3. **Fine-tune.** QLoRA on the student with PyTorch, Transformers and PEFT, on a free T4.
4. **Evaluate.** Execution accuracy on databases the student never saw in training.
5. **Ablate.** Filter strictness, dataset size, LoRA rank, sample rows in the prompt.

Training and evaluation databases are disjoint by construction, and the code
refuses to train or evaluate if they overlap.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install        # core + dev dependencies only; no torch needed
make check          # lint + tests

sqldistill schema path/to/database.sqlite   # print the schema as the model sees it
```

Extras: `pip install -e ".[train]"` for fine-tuning and local inference,
`".[teacher]"` for synthetic data generation, `".[demo]"` for the demo.

## Repository map

- [docs/PLAN.md](docs/PLAN.md): goal, experiments, milestones with acceptance criteria.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): module contracts and invariants.
- [docs/DECISIONS.md](docs/DECISIONS.md): decision log.
- `src/sqldistill/`: the package. `configs/`: experiment configuration.

## License

MIT
