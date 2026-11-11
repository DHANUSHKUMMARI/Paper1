# Experiment log

One entry per run: what ran, on which GPU, how long, compute units (CU) used, and the outcome.
Hours and CU are measured values from `runs/<run>/compute.json`, never estimates.

## Compute-unit ledger

| Date | Run | GPU | Hours | CU used | CU left (of 100) | Outcome |
|------|-----|-----|-------|---------|------------------|---------|
| 2026-11-11 | Milestone 1 build and CP0 local checks | none (sandbox CPU) | n/a | 0 | 100 | Tests, lint and audit pass; no Colab session used yet |
| 2026-11-11 | CP0 Colab check (`00_setup_and_data.ipynb`) | none (CPU runtime) | not recorded | not recorded (CPU runtime; read the balance in Colab) | not recorded | PASS |

## Runs

No training or evaluation run has been made yet.
