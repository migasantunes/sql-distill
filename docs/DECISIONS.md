# Decision log

Newest first. One entry per decision that a reader would otherwise have to guess.

## 2026-11-16 — Initial design

- **Task: text-to-SQL.** Correctness is checkable by execution, so the metric
  is objective and the synthetic data can be verified automatically.
- **Student: `Qwen/Qwen3-1.7B`.** Apache 2.0, fits a free 16 GB T4 with 4-bit
  QLoRA. `Llama-3.2-1B-Instruct` is the stretch second student.
- **Synthetic data over training databases only.** The teacher sees Spider
  train-split schemas; evaluation uses Spider dev and BIRD Mini-Dev databases.
  Splitting by database, not by question, is what makes the score mean
  "generalises to an unseen schema".
- **Spider dev is the headline, BIRD Mini-Dev the stress test.** BIRD has known
  annotation noise and needs external knowledge; a 1.7B model's score there is
  informative but should not carry the claim.
- **Own execution-accuracy implementation**, mirroring the standard definition
  (result sets equal; order matters only when the gold query has `ORDER BY`).
  Cross-check against the official evaluation scripts at M3 and record any gap.
- **SQLite only.** Both benchmarks ship SQLite databases; no server to run.
- **Light core, heavy extras.** Tests and CI run without torch so the
  correctness-critical code (execution, filtering, metrics, split guard) is
  cheap to test on every push.
