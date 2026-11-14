# Verification checkpoints (CP0–CP10)

Criteria are defined in section 9.1 of the Research Specification. The day for each checkpoint is in the 15-day execution plan.

Claude Code: after each checkpoint, fill in its entry below. Record what was run, the exact numbers, the run folder or commit, and PASS or FAIL against the criteria. Do not start the next stage while a checkpoint is FAIL. Dhanush verifies each entry and initials it.

| CP | Day | Stage | Status | Verified by |
|----|-----|-------|--------|-------------|
| CP0 | 1 | Environment | **PASS** (11 Nov 2026) | DHANUSH|
| CP1 | 2 | Data integrity and leakage | run 1 **FAIL**; run 2: criteria met, stop and ask (blockiness), fixes decided (14 Nov 2026) | |
| CP2 | 4 | Model build | pending | |
| CP3 | 5 | Training health | pending | |
| CP4 | 6 | Generalization gate | pending | |
| CP5 | 7 | Evidence faithfulness | pending | |
| CP6 | 8 | Grounded explanations | pending | |
| CP7 | 10 | Full benchmark | pending | |
| CP8 | 11 | Claims-Edit case study | pending | |
| CP9 | 14 | Ablations, statistics, human study | pending | |
| CP10 | 15 | Manuscript integrity | pending | |

## Entry template (copy once per checkpoint)

### CPn — <stage> — <date>

- **Run / commit:** `runs/...` · `git <short sha>`
- **What was run:** commands or notebook cells
- **Numbers:** each pass criterion with its measured value
- **Result:** PASS | FAIL
- **If FAIL:** fix applied, re-run result, or fallback taken (and why)
- **Verified by Dhanush:** ☐

---

### CP0 — Environment — 2026-11-11

- **Run / commit:** branch `m1-environment` · `git 45a9329` · CI run [42877556236](https://github.com/DHANUSHKUMMARI/Paper1/actions/runs/42877556236)
- **What was run:** `ruff check src tests`; `pytest -q`; `python -m eviground.utils.repo_audit --history`; `EVIGROUND_DRY_RUN=1 pytest -q --nbmake notebooks/00_setup_and_data.ipynb`. Locally in the build sandbox (Python 3.13.18, torch 2.15.0, CPU) and in GitHub Actions (Python 3.10 and 3.12, CPU).
- **Numbers, against the pass criteria of Research Specification 9.1:**

| Criterion | Measured | Result |
|-----------|----------|--------|
| All tests pass in CI | Python 3.10: success; Python 3.12: success (lint, tests, audit, notebook dry run all green) | PASS |
| All tests pass locally | 52 passed, 0 failed, 0 skipped | PASS |
| No tokens in the repository | 0 findings in 38 tracked files; 0 findings across all 5 commits (36 blobs), including inside the `.docx` | PASS |
| No dataset, checkpoint or secret file tracked | 0 forbidden files; 0 images outside `docs/`; largest tracked file 40 KB (limit 5 MB) | PASS |
| `.gitignore` works | 9 of 9 sample data, checkpoint and secret paths ignored; 3 of 3 source paths not ignored | PASS |
| Layout under `src/eviground/` | `data`, `models`, `train`, `eval`, `explain`, `baselines`, `utils` present; 19 of 19 required paths exist | PASS |
| All tests pass in a fresh Colab runtime (mount Drive, clone, `pytest`) | 52 of 52 tests passed; Drive mounted; repository audit ok (40 tracked files, 0 findings); commit `dc6fdc2` | PASS |

- **Colab run (by Dhanush), 2026-11-11 16:16 UTC:** `notebooks/00_setup_and_data.ipynb` on a CPU runtime, commit `dc6fdc2`. Record: Drive `Paper 1/runs/checkpoints/cp0_colab.json`; package versions: `requirements.lock.txt` in the same folder.

| Item | Value from `cp0_colab.json` |
|------|------------------------------|
| Status | PASS |
| Tests | passed, 52 of 52 |
| Repository audit | ok: 40 tracked files, 0 tracked-file, secret, .gitignore or structure findings (shallow clone: 1 commit, 35 blobs; full history is scanned in CI) |
| Runtime | Python 3.13.18, torch 2.14.1 (CPU build), no GPU |
| Key library versions | transformers 5.22.1, peft 0.21.2, timm 1.0.31, numpy 2.4.6, omegaconf 2.4.0 |
| Drive | mounted at `/content/drive/MyDrive/Paper 1`; free space reported by the mount: 109.7 GB |
| Dataset folders seen | 01_GenImage, 02_DiffSeg30k, 03_SID-Set, 04_BR-Gen, 05_Chameleon, 06_MMTD-Set, 07_FaceForensics++_optional, plus `Chameleon.zip` and `INVENTORY.csv` |

- **Result: PASS.** Every CP0 criterion is met in the build sandbox, in CI and in Colab.
- **Carried forward to later checkpoints:** (1) this run used a CPU runtime, so the GPU is first exercised at CP2; (2) Colab ships transformers 5.x, so the CLIP layer names targeted by LoRA are asserted at CP2; (3) `bitsandbytes`, `grad-cam` and `diffusers` inpainting are not needed yet and are installed and locked when first used; (4) whether each dataset folder is complete (`_DONE` markers, file counts) is checked at CP1, not here.
- **Not checked by this checkpoint:** the Drive copy of the downloader notebook, and whether the tokens pasted in chat on 10 Nov were rotated (only Dhanush can confirm).
- **Verified by Dhanush:** PASS

---

### CP1 — Data integrity and leakage — 2026-11-14

- **Run / commit:** Colab CPU runtime, `main` at `git a14d717`; record Drive
  `Paper 1/runs/checkpoints/cp1_data_integrity.json` (written 2026-11-14 09:28 UTC); sample grids
  and `manifest_cp1.csv.gz` in `Paper 1/runs/checkpoints/cp1/`. Wall time 161.5 min (Cells 1–5;
  `run_cp1` itself 5,715.7 s). CU not recorded.
- **What was run:** `notebooks/01_cp1_data_integrity.ipynb`, Cells 1–5 (`ensure_data('cp1',
  stage_local=True)`, `run_cp1` with 3,000 sampler crops per class, seed 42). COCO negatives were
  still prepared with the earlier rule (shorter side 1024, centre crop 1024); the DiffSeg30k-size
  rule decided on 14 Nov is not yet in this run.
- **Scanned:** 255,999 images (COCO 5,000; DiffSeg30k 34,999; GenImage SD v1.4 80,000; GenImage
  Validation 100,000; SID-Set 36,000); 0 unreadable; local copies complete (80,000 / 70,000 /
  48,000 / 5,000 files).
- **Numbers, against the pass criteria of Research Specification 9.1 (rev 49):**

| Criterion | Measured | Result |
|-----------|----------|--------|
| 0 duplicate image hashes across train / val / test | 169 file-hash groups and 169 pixel-hash groups shared across splits (test+train 149, test+val 14, train+val 6); all 20 recorded examples are GenImage SD v1.4 training "nature" (ImageNet train) = GenImage Validation "nature" (ImageNet val) images. Within-split duplicate groups (reported only): 70 | **FAIL** |
| Class counts equal the manifests | all 13 (source, split, class) counts equal (e.g. SD v1.4 40,000 + 40,000; DiffSeg30k 28,998 / 6,001; SID-Set 10,000 / 2,000 per class; COCO 5,000) | PASS |
| Metadata classifier (format, JPEG quality, crop size) ≤ 60% real vs not real | 0.5353 balanced accuracy on 7,866 harmonized crops (format 0.50, JPEG quality 0.5353, crop width 0.50, crop height 0.50); three classes 0.3821 | PASS |
| Blockiness per class: a clear class difference means stop and ask | harmonized medians: GenImage real 1.0156 / generated 1.0451; SID-Set real 1.2586 / synthetic 1.2984 / tampered 1.3267; DiffSeg30k edited 1.0222; COCO real 1.0501. Separation above 0.60: GenImage generated vs real 0.6675; SID-Set tampered vs real 0.6587; all sources (sampler mix) generated vs edited 0.6089 | **stop and ask** |

- **Near-duplicates of test images in train + val** (pHash Hamming ≤ 6; more than 1% means stop and
  ask): P1 GenImage Validation 193 of 100,000 (0.19%); P2 DiffSeg30k Glide/Kolors 33 of 2,809
  (**1.18%**, 30 of them nearest to SID-Set training tampered images); P3b SID-Set validation 353 of
  6,000 (**5.88%**, 315 nearest to SID-Set training images, 35 to DiffSeg30k); P7 Claims-Edit 1 of 150
  (0.67%, an identical SID-Set training real image). COCO negatives dropped as near-duplicates of a
  test image: 10 of 4,850.
- **Other diagnostics:** DiffSeg30k mask values {0,…,8}, PNG mode L, no size mismatch; roles: train
  pool 12,746, excluded hold-out 11,252, **excluded with an empty mask 5,000**, P2 test 2,809, unused
  validation 3,192. Harmonization: GenImage real kept 35,811, too small 4,189; SID-Set real
  recompressed to 75 / 4:2:0 8,800, kept 1,000, skipped below 75: 193, no quality: 7. Crops: 1,134 of
  9,000 draws failed the 1–90% edited share in 10 tries (DiffSeg30k 948 of 2,000 edited draws
  succeeded). Diagnostics (not criteria): original size 0.818, harmonized file bytes 0.6151, raw
  files 0.8715.
- **Result:** **FAIL** (cross-split duplicates), with two stop-and-ask flags (near-duplicates over 1%
  in P2 and P3b; blockiness class differences). Reported to Dhanush with proposed fixes; nothing
  changed until he decides.
- **Verified by Dhanush:** ☐

### CP1 run 2 — Data integrity and leakage — 2026-11-14

- **Run / commit:** Colab CPU runtime, `main` at `git c001cb2` (fixes of PR #12); record Drive
  `Paper 1/runs/checkpoints/cp1_data_integrity.json` (written 2026-11-14 10:35 UTC, replacing run 1's
  file; run 1's numbers are in the entry above). Wall time 25.7 min (scan reused from run 1).
  CU not recorded.
- **What was run:** `notebooks/01_cp1_data_integrity.ipynb`, Cells 1–5. Local copies checked file by
  file: 80,000 / 70,000 / 48,000 / 5,000 files, 0 missing, 0 other size, 0 extra.
- **Left out of training** (test sets unchanged): 28 exact duplicates of test images (GenImage → P1);
  467 near-duplicates (pHash ≤ 6): GenImage 128 → P1, 2 → P3b; DiffSeg30k 16 → P1, 58 → P2, 118 → P3b;
  SID-Set 14 → P1, 1 → P2, 117 → P3b, 1 → P7; COCO 12 → P1; 10,000 SID-Set tampered (fallback).
  DiffSeg30k unedited images in training as real negatives: 4,370 train + 486 val. COCO negatives
  at DiffSeg30k sizes: 176 unchanged, 478 cropped, 180 downscaled, 4,016 upscaled.
- **Numbers, against the pass criteria of Research Specification 9.1 (rev 49):**

| Criterion | Measured | Result |
|-----------|----------|--------|
| 0 duplicate image hashes across train / val / test | 0 file-hash and 0 pixel-hash groups across splits (within-split groups, reported only: 43) | PASS |
| Class counts equal the manifests | all 13 equal | PASS |
| Metadata classifier (format, JPEG quality, crop size) ≤ 60% real vs not real | 0.5411 on 7,426 harmonized crops (JPEG quality alone 0.5411; format, crop width, crop height 0.50); three classes 0.4294 | PASS |
| Blockiness per class: a clear class difference means stop and ask | harmonized medians: GenImage real 1.0158 / generated 1.0406; SID-Set real 1.2515 / synthetic 1.2827; DiffSeg30k real 1.0241 / edited 1.0383; COCO 1.0408. Above 0.60: GenImage generated vs real 0.6675; all sources (sampler mix) real vs edited 0.6205, generated vs edited 0.642 | **stop and ask** |

- **Near-duplicates of test images in train + val** (pHash ≤ 6): 0 in P1, P2, P3b and P7 (at ≤ 8:
  791, 50, 272, 0).
- **Diagnostics for the blockiness stop:** GenImage generated vs real separates already on the raw
  files (0.6225) and on mid-block lines of the raw files, where JPEG puts no edges (0.6575; medians
  1.0066 vs 0.9955): a trace of the generator, not of the harmonization. The pooled separations against
  "edited" come from SID-Set's quality-75 images being one third of real and generated but absent from
  edited since SID-Set tampered left training.
- **Decisions by Dhanush (14 Nov 2026):** accept GenImage's difference as a generator trace (reported,
  not a stop); compress one seeded third of DiffSeg30k edited training images at quality 75, 4:2:0, so
  quality-75 images make up one third of every class. Run 3 measures the result.
- **Other diagnostics:** metadata original size 0.7575, harmonized file bytes 0.5326, raw files
  0.7948; crops: 1,574 of 9,000 draws failed the 1–90% edited share in 10 tries (DiffSeg30k edited:
  1,426 of 3,000 succeeded).
- **Result:** criteria met; **stop and ask** on blockiness, resolved by the decisions above; CP1
  is recorded as PASS only after run 3.
- **Verified by Dhanush:** ☐

