# Experiment log

One entry per run: what ran, on which GPU, how long, compute units (CU) used, and the outcome.
Hours and CU are measured values from `runs/<run>/compute.json`, never estimates.

## Compute-unit ledger

| Date | Run | GPU | Hours | CU used | CU left (of 100) | Outcome |
|------|-----|-----|-------|---------|------------------|---------|
| 2026-11-11 | Milestone 1 build and CP0 local checks | none (sandbox CPU) | n/a | 0 | 100 | Tests, lint and audit pass; no Colab session used yet |
| 2026-11-11 | CP0 Colab check (`00_setup_and_data.ipynb`) | none (CPU runtime) | not recorded | not recorded (CPU runtime; read the balance in Colab) | not recorded | PASS |
| 2026-11-12 | Download 1 (`00c`, commit `b3f304a`): genimage_val, diffseg30k, sidset | none (CPU runtime) | 0.366 | not recorded (`CU_PER_HOUR` not set) | not recorded | DiffSeg30k downloaded and verified; SID-Set verified in place; GenImage Validation: 5 of 8 generators |
| 2026-11-12 | Download 2 (`00c`, commit `ad3b07d`, reused runtime) | none (CPU runtime) | 0.023 | not recorded | not recorded | Both failed: old code still in memory; Kaggle secret not visible |
| 2026-11-12 | Download 3 (`00c`, commit `ad3b07d`): genimage_val, genimage_sd14 | none (CPU runtime) | 0.534 | not recorded | not recorded | Validation fill wrote 3 tar files (36,000 images), count 96,411 vs 100,000 expected; SD v1.4 archive downloaded (96.5 GB) but subset stopped by the wrong "Drive free" figure (machine disk) |
| 2026-11-13 | Download 4 (`00c`, commit `74e08af`): genimage_sd14 | none (CPU runtime) | 0.186 (since the last run of Cell 2) | not recorded | not recorded | SD v1.4 subset verified: 80,000 images (40,000 + 40,000), 8 tar files, 22.9 GB; local archive deleted |

## Runs

No training or evaluation run has been made yet.

### Data downloads (Milestone 2), 12–13 Nov 2026

Hours are the `wall_time_h` of each run record in Drive `Paper 1/runs/checkpoints/cp1_download_*.json`
(time from Cell 2 to Cell 4 of `00c_data_download.ipynb`). CU were not recorded: the runs used CPU
runtimes and `CU_PER_HOUR` was left empty; read the balance in Colab to fill this in.

- The run records' "Drive free" figures before 13 Nov are the Colab machine's disk, not the Drive
  quota (see `docs/TESTING.md`, 12 Nov). Drive usage on 12 Nov: 134.28 GB of 5 TB (read by Dhanush).
- State after download 4: DiffSeg30k, SID-Set and the SD v1.4 subset verified (`_DONE`); GenImage
  Validation pending (tar-to-loose conversion of the three filled generators and the count check).
