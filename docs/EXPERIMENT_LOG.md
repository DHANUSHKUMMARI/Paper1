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
| 2026-11-14 | CP1 run 2 (`01`, commit `c001cb2`), Cells 1–5 | none (CPU runtime) | 0.429 (25.7 min) | not recorded | not recorded | Criteria met (0 cross-split duplicates; counts equal; metadata 0.5411); stop and ask on blockiness (GenImage 0.6675, pooled vs edited 0.6205 / 0.642) |
| 2026-11-14 | CP1 run 3 (`01`, commit `2feec5a`), Cells 1–5 | none (CPU runtime) | 0.359 (21.5 min) | not recorded | not recorded | **CP1 PASS**: 0 cross-split duplicates; counts equal; metadata 0.5089; no blockiness stop (GenImage trace accepted) |
| 2026-11-14 | CP1 run 4 (`01`, commit `0f8b35c`), Cells 1–5 | none (CPU runtime) | 1.169 (70.1 min) | not recorded | not recorded | **CP1 PASS** (spec-review design): 0 cross-split duplicates; counts equal; metadata 0.5055; no stop; `.jsonl` keys only file names; q75 share 0.363 / 0.370 / 0.365 |
| 2026-11-14 | Crop diagnostic (`02`, commit `b087db6`), Cells 1–4 | none (CPU runtime) | 0.343 (20.6 min) | not recorded | not recorded | Run 4 reproduced (1,585 of 3,000 edited draws failed); 1,583 failed above 90%, 2 below 1%; 5,930 of 11,382 edited training images have no valid window; median edited share 100% (1,048,576 px) |
| 2026-11-14 | CP1 run 5 (`01`, commit `9a134bc`), Cells 1–5 | none (CPU runtime) | 1.478 (88.7 min) | not recorded | not recorded | **CP1 PASS** (accepted): criteria met (metadata 0.5071); stop flag base disagreement 13.1% accepted by Dhanush; P2 2,434 (1,287 real-base); 68 of 3,000 edited draws failed, all replaced |
| 2026-11-15 | Pack training shards (`04`, commit `7c7dcfb`), Cells 1–5 | none (CPU runtime) | 1.239 (74.3 min; about 44 min of it staging the four sources) | 0.10 (1.239 h × 0.08 CU/h, the rate shown in Colab's resource panel) | 88 (balance read in Colab after the run) | 116,759 images in 18 shards, 15.75 GB (estimate 16.30 GB); 11,779 edit masks; packed bytes equal a fresh harmonization for 800 / 800 sampled images (200 per source); index sha256 `1085a527…` |
| 2026-11-15 | CP2 attempts (`05`, commits `7c7dcfb`–`911bf37`), stopped | L4 | not recorded (no run record: the notebook stopped before Cell 9) | not recorded (1.7 CU/h; read the balance in Colab) | not recorded | Stopped in Cell 4: (1) Drive-cached CLIP file was 79 bytes of link text; (2) Colab's torchao 0.10 refused by peft; (3) an old copy of the notebook's cells pointed at the Drive cache again, caught by the new weights hash check; (4) the epoch plan read ~9,000 edit masks through the Drive mount (>40 min, interrupted). Fixes: PRs #24 and the local shard copy |
| 2026-11-15 | CP2 (`05`, commit `2673e8f`), Cells 1–9 | L4 | 0.246 (Cells 2–9) | 0.42 (1.7 CU/h) | not recorded (read the balance in Colab) | **CP2 PASS** (script verdict): overfit 100% at step 25; kill-resume max loss difference 0.024%; views 0.064 ms, trace 0.631 ms per image; peak 5.64 GB at batch 32 bf16; loader 253.9 images/s; shards staged in 404 s |
| 2026-11-15 | CP3 unified, seed 42 (`06`, commit `f2d0e1c`): A1 + 20 epochs | L4 | 3.015 (session; 2.056 GPU training hours) | 5.85 (1.94 CU/h) | not recorded (read the balance in Colab) | **CP3 PASS** (script verdict): A1 accuracy 1.0 at step 25, mask F1 0.871 at step 50; best epoch 20: macro AUC 0.998221, mask F1 0.762032, real marks 0.0815 / area 0.008319, generated 0.0179 / 0.004010, binary mIoU 0.661394 |
| 2026-11-16 | CP3 detection, seed 42 (`06`, commit `f2d0e1c`): 15 epochs | L4 | 1.547 (session; 1.023 GPU training hours) | 3.00 (1.94 CU/h) | not recorded (read the balance in Colab) | **CP3 PASS** (script verdict): best epoch 15: binary AUC 0.999717, SD v1.4 accuracy 0.991568 |
| 2026-11-17 | Localization diagnosis, CP3 unified (`08`, commit `c8ed936`), validation only | L4 | 0.827 | 1.60 (1.94 CU/h) | not recorded | Whole-image edited IoU 0.3946 (crops 0.677); over-marking 28.9% of pixels (AI bases 35.9%), misses 19.1%; real images called edited 8.9% |
| 2026-11-17 | CP4 fix sampler check (`10`, commit `616bec5`) | none (CPU runtime) | 0.082 (294 s) | not recorded (CPU runtime) | not recorded | No material change: real vs not real 0.5122 → 0.5120; every change ≤ 0.002 |
| 2026-11-17 | CP4 gate, CP3 models, P2 off (`09`, commit `616bec5`), P1 development split | L4 | 1.405 | 2.73 (1.94 CU/h) | not recorded | **G1 FAIL** (detection 0.7846 raw / 0.7771 harmonized; probe 0.8136 / 0.8015); RF PASS; G2 not run |
| 2026-11-17 | CP4 fix: unified seed 42 with context crops (`11`, commit `ff36b71`): 20 epochs | L4 | 3.056 (session; 2.0126 GPU training hours) | 5.20 session, of which training 3.421 (1.7 CU/h) | not recorded | Health FAIL on A2 only (real-crop marked area 0.010302 > 0.01); best epoch 15: macro AUC 0.997864, mask F1 0.758295 |
| 2026-11-17 | Localization diagnosis, fixed unified (`08`, `RUN_EXPERIMENT='eviground_ctx'`, commit `ff36b71`) | L4 | 0.652 | 1.26 (1.94 CU/h) | not recorded | Whole-image edited IoU 0.4335; over-marking 21.5% (AI bases 26.3%); misses 22.8%; real images called edited 10.5% |
| 2026-11-18 | CP4 gate with P2 on, **by mistake on the CP3 unified model** (`09`, `RUN_EXPERIMENT` left `None`, commit `33707a3`) | L4 | 2.232 | 4.33 (1.94 CU/h) | not recorded | Not the gate run (G2 is gated on the fixed model). CP3 unified on P2 dev: edited IoU 0.4904; G1/RF identical to 17 Nov |
| 2026-11-18 | **CP4 G2 gate**, fixed unified, P2 only (`09`, `RUN_EXPERIMENT='eviground_ctx'`, `P1_ENABLED=False`, commit `ff1b14d`) | L4 | 0.370 | 0.72 (1.94 CU/h) | about 44.3 (45 reported by Dhanush before the run, minus 0.72) | **G2 FAIL**: edited IoU 0.5101 (real base 0.7471; held-out pixels 0.4588). CP4 FAIL (G1, G2); RF PASS |
| 2026-11-18 | Frozen evaluation subsets (`12`, CPU, commit after PR #46) | none (CPU runtime) | 0.0977 | 0.008 (0.08 CU/h) | about 44.3 | Chameleon.zip 2,683,359,341 bytes, 26,033 images (real 14,863, generated 11,170), 0 unreadable; P4 2,500 + 2,500 (`50167dd2…`); P1 20,000 = 1,250 per generator and label (`881098bb…`) |
| 2026-11-18 | Unified, final configuration, seeds 123 and 2026 (`13`, commit `911e680`), one session | L4 | 5.768 (session; training 2.0364 + 1.7304 GPU h) | **about 12.5–13 measured from the balance** (44.3 → 31–32 reported by Dhanush); the notebook recorded 9.805 at 1.7 CU/h (runs 3.462 + 2.942), so the real rate is about 2.2 CU/h | about 31–32 | Seed 123 health **PASS** (best epoch 18; macro AUC 0.997629; mask F1 0.743497; A2 area 0.008986). Seed 2026 health **FAIL on A2 only** (area 0.010192 > 0.01; early stop at epoch 17, best 12; macro AUC 0.997456; mask F1 0.754473); kept, as seed 42 was (Dhanush, 18 Nov) |
| 2026-11-18 | Frozen P6 subsets (`12b`, CPU, commit `a843a74`) | none (CPU runtime) | 0.182 | 0.015 (0.08 CU/h) | about 31–32 | P6 1,500 (`c12ba888…`), P6_mllm 500 (`1aa9bb7e…`), P6_deletion 300 (`c42ed65c…`); real images GenImage 50 / SID-Set validation real 50 (no unedited DiffSeg30k validation image exists) / Chameleon 50; 0 near-duplicates of training |
| 2026-11-19 | CP5 + P6 explanation pass (`14`, commit `1973329`), unified final configuration seed 42 | L4 | 0.639 (last session only) | **about 7 by balance** (the record: 1.405 at 2.2 CU/h, last session only; an earlier session went unrecorded; estimate 3.0) | 28.19 | **CP5 FAIL**: Grad-CAM++ EPG 1.38× (bar 1.5×), SSIM 0.455; decoder EPG 2.00×, SSIM 0.545; pixel deletion confounded (≈ insertion). Fallback: decoder primary. P6 EPG ratio Grad-CAM++ 1.16× / decoder 1.65× |
| 2026-11-24 | CP5 repair: token deletion + randomization rerun, P6 cache built (`15`, commit `1050c24`), unified final configuration seed 42 | **T4** (the record says `Tesla T4`; planned L4) | 1.3405 | **1.71 by balance** (28.19 → 26.48, Dhanush); the record says 2.949 at the L4 rate of 2.2 CU/h, which does not apply to a T4; its `cu_balance_start` reads 28.02 | 26.48 | Token deletion **VALID** (oracle 0.511 vs random 0.569, p 5.3e-6); decoder **PASS** (0.435, p 2.5e-35); Grad-CAM++ **PASS** (0.503, p 1.7e-17). Randomization last step, validation: Grad-CAM++ SSIM 0.376, decoder 0.546 (fail; Spearman 0.130). 26 SID-Set masks stretched to their image (aspect mismatch) |

## Runs

Training runs so far (seed 42, notebook 06, commit `f2d0e1c`); the numbers per criterion are in
`docs/CHECKPOINTS.md` (CP3), the compute in the table above:

- `Paper 1/runs/eviground_unified_seed42/` (A1 record in `eviground_unified_seed42_overfit/`);
- `Paper 1/runs/eviground_detection_seed42/`.

The CP4 fix run (notebook 11, `data.context_crop_share=0.3333`, commit `ff36b71`):
`Paper 1/runs/eviground_ctx_unified_seed42/` (A1 record reused from CP3).

The extra seeds of the final configuration (notebook 13, commit `911e680`, 18 Nov 2026; A1 record reused from CP3);
CP3 health per seed in `docs/CHECKPOINTS.md`:

- `Paper 1/runs/eviground_ctx_unified_seed123/`;
- `Paper 1/runs/eviground_ctx_unified_seed2026/`;
- the health files `training_health_unified_ctx_seed{123,2026}.json` and the session record `compute_unified_seeds.json`
  in `Paper 1/runs/checkpoints/`.

Frozen evaluation subsets: P1 and P4 (notebook 12) and P6, P6_mllm, P6_deletion (notebook 12b), all in
`Paper 1/runs/checkpoints/eval_subsets.json`; no model has been scored on them yet.

Test-set evaluation so far: only the CP4 development splits, for the gate. P1 dev was scored on the CP3 models (notebook 09,
17 Nov 2026), and P2 dev on the fixed unified model (G2, 18 Nov 2026; and once on the CP3 unified model by mistake, as
recorded above).

### Training shards (Milestone 3), 15 Nov 2026

From Drive `Paper 1/runs/checkpoints/pack_shards.json` (notebook 04, commit `7c7dcfb`):

- Manifest: CP1 run 5 (`manifest_sha256` `ef4f4f12…`); mask rule "diffseg30k: AI-generated base read
  as 0 (border + most frequent value), v1, 14 Nov 2026"; `index_sha256` `1085a527…`.
- 116,759 images (train / val): COCO real 4,340 / 483; DiffSeg30k edited 10,602 / 1,177 and real
  4,344 / 479; GenImage generated 35,937 / 3,993 and real 32,139 / 3,597; SID-Set generated 8,890 / 988
  and real 8,804 / 986. Edit masks: 11,779 (every edited image, no other).
- 18 shards, 15.75 GB on Drive (estimate 16.30 GB: 3.5% above); Drive quota free before: 5,250.8 GB.
- Check: packed bytes equal a fresh harmonization (and the stored edit mask the CP1 mask) for 200
  seeded images of each source: 0 mismatches.
- The session record's `drive_free_gb` (109.8) is the Colab file system's figure, not the quota.

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
