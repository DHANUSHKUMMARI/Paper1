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
| 2026-11-13 | Download 5 (`00c`, commit `dc87494`): genimage_val | none (CPU runtime) | 0.385 | not recorded | not recorded | SD v1.4 fill tar found unreadable, 12,000 images refilled from the archive (25.4 GB download, local only); count 96,411: Midjourney 6,000 real / 2,411 generated |
| 2026-11-13 | Download 6 (`00c`, after PR #7): genimage_val | none (CPU runtime) | 0.653 (39.2 min) | not recorded | not recorded | 3,589 missing Midjourney images added; Drive equals the archive counts; **verified: 100,000 images, 8 generators** |
| 2026-11-13 | Data probe (`00b`, commit `0c6d6d4`), read-only | none (CPU runtime) | 0.869 (52.1 min) | not recorded | not recorded | `cp1_probe*.json` in `runs/checkpoints`: formats, JPEG qualities, sizes, blockiness per source and class; DiffSeg30k card (mask map 0–8) |
| 2026-11-14 | CP1 run (`01`, commit `a14d717`), Cells 1–5 | none (CPU runtime) | 2.692 (161.5 min) | not recorded | not recorded | **CP1 FAIL**: 169 cross-split duplicate groups; near-duplicates P2 1.18%, P3b 5.88%; blockiness separations 0.6675 (GenImage) and 0.6587 (SID-Set); metadata classifier 0.5353 |

## Runs

No training or evaluation run has been made yet.

### Data downloads (Milestone 2), 12–13 Nov 2026

Hours are the `wall_time_h` of each run record in Drive `Paper 1/runs/checkpoints/cp1_download_*.json`
(time from Cell 2 to Cell 4 of `00c_data_download.ipynb`). CU were not recorded: the runs used CPU
runtimes and `CU_PER_HOUR` was left empty; read the balance in Colab to fill this in.

- The run records' "Drive free" figures before 13 Nov are the Colab machine's disk, not the Drive
  quota (see `docs/TESTING.md`, 12 Nov). Drive usage on 12 Nov: 134.28 GB of 5 TB (read by Dhanush).
- State after download 6 (13 Nov 2026): all four CP1 datasets verified (`_DONE`): GenImage
  Validation (100,000 images; per generator 6,000 real + 6,000 generated, SD v1.5 8,000 + 8,000, as in
  the Kaggle archive), GenImage SD v1.4 subset (40,000 + 40,000), DiffSeg30k (both zips, SHA-256),
  SID-Set (10,000 / 2,000 per class). Drive quota after the run: 244.0 GB used of about 5.5 TB.
- Leftover: `01_GenImage/Validation/genimage_val_sd14.tar` (3.2 GB, unreadable) and its `.json`; SD v1.4
  was refilled as loose files, so both can be deleted.
- Data probe (13 Nov 2026, `cp1_probe_summary.json`, 300 headers and 40 blockiness samples per
  group): see the CP1 findings in `docs/CHECKPOINTS.md` once CP1 is recorded. Open questions raised
  with Dhanush: SID-Set compression history differs by class; DiffSeg30k has no JPEG images to
  measure a harmonization quality from; source image size versus crop size in the metadata check;
  `Chameleon.zip` not found.
