# Verification checkpoints (CP0–CP10)

Criteria are defined in section 9.1 of the Research Specification. The day for each checkpoint is in the 15-day execution plan.

Claude Code: after each checkpoint, fill in its entry below. Record what was run, the exact numbers, the run folder or commit, and PASS or FAIL against the criteria. Do not start the next stage while a checkpoint is FAIL. Dhanush verifies each entry and initials it.

| CP | Day | Stage | Status | Verified by |
|----|-----|-------|--------|-------------|
| CP0 | 1 | Environment | **PASS** (11 Nov 2026) | DHANUSH|
| CP1 | 2 | Data integrity and leakage | **PASS** (14 Nov 2026, run 5, DiffSeg30k mask reading; runs 1–4 below) | DHANUSH|
| CP2 | 4 | Model build | **PASS** (15 Nov 2026, notebook 05 on an L4; script verdict, awaiting Dhanush) |DHANUSH |
| CP3 | 5 | Training health | **PASS** (seed 42, both models; notebook 06 on an L4, 15–16 Nov 2026; script verdict, awaiting Dhanush) |DHANUSH |
| CP4 | 6 | Generalization gate | **FAIL** (G1 and G2; RF PASS; one fix used, no second fix; 17–18 Nov 2026; awaiting Dhanush's decision on the narrowed claims) | |
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

### CP1 run 3 — Data integrity and leakage — 2026-11-14 — **PASS**

- **Run / commit:** Colab CPU runtime, `main` at `git 2feec5a` (decisions of run 2 merged in PR #13);
  record Drive `Paper 1/runs/checkpoints/cp1_data_integrity.json` (written 2026-11-14 11:19 UTC;
  replaces the run-2 file, whose numbers are in the entry above); sample grids and
  `manifest_cp1.csv.gz` in `Paper 1/runs/checkpoints/cp1/`. Wall time 21.5 min (scan reused).
  CU not recorded.
- **What was run:** `notebooks/01_cp1_data_integrity.ipynb`, Cells 1–5. Local copies checked file by
  file: 80,000 / 70,000 / 48,000 / 5,000 files, 0 missing, 0 other size, 0 extra. 255,999 images
  indexed, 0 unreadable.
- **Left out of training** (test sets unchanged): 28 exact and 467 near duplicates of test images,
  10,000 SID-Set tampered. DiffSeg30k edited harmonization: 8,431 with the GenImage real
  distribution, 4,220 at quality 75 / 4:2:0 (the seeded third), 47 too small.
- **Numbers, against the pass criteria of Research Specification 9.1 (rev 49):**

| Criterion | Measured | Result |
|-----------|----------|--------|
| 0 duplicate image hashes across train / val / test | 0 file-hash and 0 pixel-hash groups across splits (within-split groups, reported only: 43) | PASS |
| Class counts equal the manifests | 13 of 13 equal | PASS |
| Metadata classifier (format, JPEG quality, crop size) ≤ 60% real vs not real | 0.5089 on 7,426 harmonized crops (JPEG quality alone 0.5089; format, crop width, crop height 0.50); three classes 0.3451 | PASS |
| Blockiness per class: a clear class difference means stop and ask | harmonized medians: GenImage real 1.0158 / generated 1.0406; SID-Set real 1.2515 / synthetic 1.2827; DiffSeg30k real 1.0241 / edited 1.0765; COCO 1.0408. Separations: DiffSeg30k edited vs real 0.5904; SID-Set synthetic vs real 0.5825; sampler mix real vs not real 0.5472, real vs generated 0.5917, real vs edited 0.4675, generated vs edited 0.5717. No clear difference; GenImage generated vs real 0.6675 reported as accepted (generator trace, decision of 14 Nov 2026) | PASS (no stop) |

- **Near-duplicates of test images in train + val** (pHash ≤ 6): 0 in P1, P2, P3b and P7 (at ≤ 8:
  791, 50, 272, 0). DiffSeg30k mask values {0,…,8}.
- **Diagnostics (not criteria):** metadata with subsampling 0.512, original size
  0.7575, harmonized file bytes 0.5461, raw files 0.7948; crops: 1,574 of 9,000 draws
  failed the 1–90% edited share in 10 tries (DiffSeg30k edited: 1,426 of 3,000 succeeded); COCO
  negatives at DiffSeg30k sizes: 176 unchanged, 478 cropped, 180 downscaled, 4,016 upscaled.
- **Result:** **PASS**. All three criteria met and no stop flag (the GenImage generator trace is
  accepted and reported).
- **Verified by Dhanush:** ☐

### CP1 run 4 — Data integrity and leakage — 2026-11-14 — **PASS**

- **Run / commit:** Colab CPU runtime, `main` at `git 0f8b35c` (spec-review design merged in PR #15);
  record Drive `Paper 1/runs/checkpoints/cp1_data_integrity.json` (written 2026-11-14 15:54 UTC;
  replaces the run-3 file, whose numbers are in the entry above); sample grids and
  `manifest_cp1.csv.gz` in `Paper 1/runs/checkpoints/cp1/`. Wall time 70.1 min (Cells 1–5, copies
  made again; `run_cp1` 23.9 min). CU not recorded.
- **What changed since run 3** (data/README decisions 13–14): DiffSeg30k unedited images left out
  of training; the same seeded third of DiffSeg30k (all classes) and COCO compressed at quality 75,
  4:2:0; real shares GenImage 1/3, SID-Set 1/6, COCO 1/2.
- **What was run:** `notebooks/01_cp1_data_integrity.ipynb`, Cells 1–5. Local copies checked file by
  file: 80,000 / 70,000 / 48,000 / 5,000 files, 0 missing, 0 other size, 0 extra. 255,999 images
  indexed, 0 unreadable.
- **Left out of training** (test sets unchanged): 28 exact and 323 near duplicates of test images,
  10,000 SID-Set tampered, 5,000 DiffSeg30k unedited. Harmonization: COCO 3,208 with the GenImage
  real distribution, 1,604 at quality 75 / 4:2:0 (seeded third), 26 too small; DiffSeg30k edited
  8,431 / 4,220 / 47.
- **Numbers, against the pass criteria of Research Specification 9.1 (rev 49):**

| Criterion | Measured | Result |
|-----------|----------|--------|
| 0 duplicate image hashes across train / val / test | 0 groups across splits | PASS |
| Class counts equal the manifests | 13 of 13 equal | PASS |
| Metadata classifier (format, JPEG quality, crop size) ≤ 60% real vs not real | 0.5055 on 7,415 harmonized crops (JPEG quality alone 0.5055; format, crop width, crop height 0.50); three classes 0.3418 | PASS |
| Blockiness per class: a clear class difference means stop and ask | harmonized medians: GenImage real 1.0201 / generated 1.0406; SID-Set real 1.261 / synthetic 1.2827; COCO 1.0878; DiffSeg30k edited 1.0768. Separations: real (COCO) vs DiffSeg30k edited 0.4911; SID-Set synthetic vs real 0.545; sampler mix real vs not real 0.5155, real vs generated 0.5537, real vs edited 0.4539, generated vs edited 0.5183. DiffSeg30k edited vs real: not applicable (no DiffSeg30k reals). GenImage generated vs real 0.6325 reported as accepted (generator trace) | PASS (no stop) |

- **Stop flags:** all false (blockiness, near-duplicates, mask values outside 0–8, DiffSeg30k
  `.jsonl` unexpected keys, `.jsonl` census incomplete, images not scanned).
- **DiffSeg30k `.jsonl` census (every line):** `train_copy/metadata.jsonl` 28,998 lines,
  `validation_copy/metadata.jsonl` 6,001 lines; keys `image_file_name` and `mask_file_name` on
  every line, nothing else; 0 unparsable. No field gives the base type.
- **Image-size indication (not a field):** the 5,000 unedited training images all have both
  sides ≤ 640 px (COCO size); the 12,698 edited training-pool images: 6,389 at 1024×1024, 6,309
  with both sides ≤ 640. Dhanush kept the unedited images out (14 Nov 2026).
- **Quality 75, 4:2:0 as sampled:** real 0.363, generated 0.3703, edited 0.3654 (target one third
  plus GenImage's own quality-75 files). By source: COCO 0.366, GenImage real 0.042, GenImage
  generated 0.0555, SID-Set real 0.996, SID-Set synthetic 1.0, DiffSeg30k edited 0.3654.
- **Near-duplicates of test images in train + val** (pHash ≤ 6): 0 in P1, P2, P3b and P7 (at ≤ 8:
  778, 18, 260, 0). DiffSeg30k mask values {0,…,8}.
- **Diagnostics (not criteria):** metadata with subsampling 0.5055, original size 0.7701,
  harmonized file bytes 0.5527, raw files 0.8843; raw-file blockiness separation GenImage 0.5725,
  SID-Set 0.8025; mid-block raw GenImage 0.6775. Crops: 1,585 of 3,000 edited draws failed the
  1–90% edited share (1,415 succeeded; Milestone 2 loader proposal pending). COCO negatives at
  DiffSeg30k sizes: 176 unchanged, 478 cropped, 180 downscaled, 4,016 upscaled.
- **Result:** **PASS**. All three criteria met and no stop flag (the GenImage generator trace is
  accepted and reported).
- **Verified by Dhanush:** ☐

### CP1 run 5 — Data integrity and leakage — 2026-11-14 — **PASS** (accepted by Dhanush)

- **Run / commit:** Colab CPU runtime, `main` at `git 9a134bc` (mask reading merged in PR #18); record
  Drive `Paper 1/runs/checkpoints/cp1_data_integrity.json` (written 2026-11-14 19:51 UTC; replaces the
  run-4 file); grids and `manifest_cp1.csv.gz` in `Paper 1/runs/checkpoints/cp1/`. Wall time 88.7 min
  (`run_cp1` 26.4 min). CU not recorded.
- **What changed since run 4** (data/README decisions 15–17): an AI-generated base's mask value is
  read as 0 (border majority = most frequent value; disagreements leave training); Glide / Kolors
  out of training wherever they appear, P2 = validation images where they made an edit;
  DiffSeg30k's unedited images are real negatives again (real = 1/3, 1/6, 1/4, 1/4); edited crops
  with the training rule, failed draws replaced.
- **Numbers, against the pass criteria of Research Specification 9.1:**

| Criterion | Measured | Result |
|-----------|----------|--------|
| 0 duplicate image hashes across train / val / test | 0 groups across splits | PASS |
| Class counts equal the manifests | 13 of 13 equal | PASS |
| Metadata classifier (format, JPEG quality, crop size) ≤ 60% real vs not real | 0.5071 on 9,000 harmonized crops (JPEG quality alone 0.5071; format, crop width, crop height 0.50); three classes 0.3469 | PASS |
| Blockiness per class: a clear class difference means stop and ask | harmonized medians: GenImage real 1.0201 / generated 1.0406; SID-Set real 1.2702 / synthetic 1.2827; COCO 1.0746; DiffSeg30k real 1.0815 / edited 1.0913. Separations: DiffSeg30k edited vs real 0.4906; real (COCO + DiffSeg30k) vs DiffSeg30k edited 0.4745; SID-Set 0.5375; sampler mix real vs not real 0.5429, real vs generated 0.5579, real vs edited 0.5086, generated vs edited 0.5274. GenImage generated vs real 0.6325 reported as accepted (generator trace) | PASS (no stop) |

- **Stop flag raised:** `diffseg_base_unclear_over_5pct` (a limit I added in PR #18, not a spec 9.1
  criterion): border and pixel count disagree on the base for 2,048 of 15,687 full masks (13.1%).
  By role: 645 left out of training (`excluded_base_unclear`), 1,016 held out anyway, 235 in P2,
  152 unused validation. **Dhanush accepted run 5 as PASS** (14 Nov 2026); the disagreements stay
  out of training and counted. The accepted share is in `CP1Config.base_unclear_accepted` (0.1306),
  so a rerun on the same data does not stop again; a higher share still does. All other stop flags
  false.
- **Mask readings:** agree 13,033, disagree 2,048, no edit left 606, real base 14,312, unedited
  5,000. Training pool: 5,560 AI-base edited, 6,266 real-base edited, 4,856 unedited (real
  negatives); left out: 645 base unclear, 228 no edit, 11,252 hold-out.
- **P2:** 2,434 validation images where Glide or Kolors made an edit: 1,287 real-base (the subset
  compared with the DiffSeg30k paper), 1,147 AI-base (235 of them with a base disagreement); 375
  images with Glide/Kolors only as the base are left out.
- **Valid crops (read masks):** 21,477 of 28,998 train-split images; 11,639 of 11,826 edited
  training images.
- **Edited crops (training rule):** 68 of 3,000 first-pass draws failed (65 every try above 90%, 3
  below 1%); shares of their tries: (0.1–0.5%] 12, (0.5–1%) 18, (90–99%] 99, (99–100%) 204, 100% 347;
  69 failed draws in all, replaced in 2 rounds, 0 unresolved. The 1% minimum stays.
- **Quality 75, 4:2:0 as sampled:** real 0.3643, generated 0.3703, edited 0.377 (COCO 0.3533,
  DiffSeg30k real 0.384, DiffSeg30k edited 0.377).
- **Left out of training** (test sets unchanged): 28 exact and 466 near duplicates of test images,
  10,000 SID-Set tampered. Near-duplicates at ≤ 6: 0 in P1, P2, P3b and P7.
- **Diagnostics:** metadata original size 0.7868, harmonized file bytes 0.5232, raw files 0.7685;
  COCO negatives at DiffSeg30k sizes: 204 unchanged, 490 cropped, 189 downscaled, 3,967 upscaled.
- **Result:** **PASS** (criteria met; the base-disagreement stop accepted by Dhanush).
- **Verified by Dhanush:** DHANUSH


---

### CP2 — Model build — 2026-11-15 — **PASS** (script verdict; awaiting Dhanush)

- **Run / commit:** `notebooks/05_cp2_model_build.ipynb`, commit `2673e8f` (main after PR #25), NVIDIA L4
  (23.7 GB), torch 2.14.1+cu130, Python 3.13.16. Record: Drive `Paper 1/runs/checkpoints/cp2_model_build.json`
  (2026-11-15 17:20 UTC); run folders `Paper 1/runs/cp2/` (`overfit`, `resume_ref`, `resume_cut`,
  `validation`).
- **Inputs:**
  - CLIP `openai/clip-vit-large-patch14` at commit `32bd6428…`; weights SHA-256 `a2bf730a…`, checked
    before loading.
  - Shards: 116,759 images, index sha256 `1085a527…`, manifest `ef4f4f12…`, mask rule v1 of 14 Nov.
    They were copied to `/content/data/shards` (18 shards, 15.75 GB, 404 s).
- **What was run:** Cells 1–9 (Run all, fresh session), `CU_PER_HOUR` 1.7.
- **Numbers, against the pass criteria (Research Specification 9.1 and Implementation Brief):**

| Criterion | Measured | Result |
|-----------|----------|--------|
| Output shapes as in section 8 (input 2 × 256²) | logits (2, 3); mask logits (2, 1, 256, 256); α (2, 256); g (2, 1); Z (2, 256, 512) | PASS |
| M2 trainable parameters < 5 M | 655,872 (LoRA 131,072, projection 524,800). Model total 322,706,209: frozen CLIP 303,179,776, trainable 19,526,433 (M3 3,653,532, M4 fusion 13,138,945, heads 2,078,084) | PASS |
| Frozen CLIP weights unchanged after training steps | SHA-256 `cdd5e4900ac8dc4d…` before and after the overfit run (the same value as the sandbox CPU build of 15 Nov) | PASS |
| 64-image overfit: ≥ 99% train accuracy within 300 steps, augmentation and dropout off | 64 images (21 real, 21 generated, 22 edited): accuracy 1.000 at the first evaluation, step 25 (total loss 1.120); 0.32 min | PASS |
| Killed run resumes at the same step, loss within 1% | killed after step 30 (exit 137), resumed from the step-20 checkpoint, finished at step 40; same batches at every step; loss difference max 0.024%, mean 0.0088% | PASS |
| Views < 5 ms per image | 0.064 ms (0.074 ms with channels_last), batch 32, bf16 | PASS |
| Trace branch < 15 ms per image | 0.631 ms (0.614 ms with channels_last) | PASS |
| Batch 32 at 256² in bf16 fits | peak 5.64 GB of 23.7 GB (5.66 GB with channels_last); training step 0.466 s (0.459 s with channels_last) | PASS |
| Loader ≥ 150 images/s | 253.9 images/s, 4 workers, from the local copy | PASS |
| Validation pass (≤ 0.5 CU) | 11,689 images in 1.78 min | PASS |

- **Untrained baseline on validation** (expected near chance):
  - macro one-vs-rest AUC 0.523: real 0.513, generated 0.545, edited 0.512;
  - mask F1 per image 0.216, pooled F1 0.234, IoU 0.133;
  - every real crop has some predicted edited pixels; mean predicted area 0.141;
  - gate mean 0.452;
  - selection score 0.370.
- **Data the model trains on:**
  - pools: real GenImage 32,139, SID-Set 8,804, COCO 4,340, DiffSeg30k 4,344; generated GenImage 35,937,
    SID-Set 8,890; edited DiffSeg30k 10,476;
  - 126 training and 14 validation edited images have no valid crop and are left out;
  - 24,000 crops per epoch.
- **Compute:** 0.246 h of wall time (Cells 2–9) at 1.7 CU/h = **0.42 CU**.
- **Notes:**
  - The four earlier attempts of 15 Nov stopped in Cell 4 (see `docs/EXPERIMENT_LOG.md`). The fixes are
    PRs #24 (CLIP cache on the local disk, torchao, pinned commit and weights hash) and #25 (local copy
    of the shards).
  - PyTorch warns that the flash-attention backward pass is not deterministic. The resumed run's losses
    stay within 0.024% of the reference.
  - channels_last stays **off** (Dhanush, 15 Nov). The training step was 1.5% faster (0.459 vs
    0.466 s), but the views stage was 16% slower in the same measurement (0.074 vs 0.064 ms per
    image), so the difference is within noise.
  - The run passed the CLIP commit as an override, which is redundant now that `configs/base.yaml`
    pins it; notebook 05 no longer adds it. Colab ran Python 3.13, so 3.13 joins CI.
- **Result:** **PASS** on every criterion (script verdict).
- **Verified by Dhanush:** DHANUSH

---

### CP3 — Training health, seed 42 — 2026-11-16 — **PASS** (script verdict; awaiting Dhanush)

- **Run / commit:** `notebooks/06_cp3_training.ipynb`, commit `f2d0e1c` (main after PR #28), branch `main`,
  NVIDIA L4 (23.7 GB), torch 2.14.1+cu130, Python 3.13.16, `CU_PER_HOUR` 1.94.
  - Unified: run folder Drive `Paper 1/runs/eviground_unified_seed42/`; A1 record
    `Paper 1/runs/eviground_unified_seed42_overfit/overfit_a1.json`; result
    `Paper 1/runs/checkpoints/cp3_training_health_unified_seed42.json` (2026-11-15 22:25 UTC).
  - Detection: run folder `Paper 1/runs/eviground_detection_seed42/`; result
    `Paper 1/runs/checkpoints/cp3_training_health_detection_seed42.json` (2026-11-16 07:09 UTC).
- **Inputs (both runs):** CLIP `openai/clip-vit-large-patch14` at `32bd6428…` (frozen-weights SHA-256
  `cdd5e4900ac8…`, as in CP2); shard index `1085a527…`, manifest `ef4f4f12…`, mask rule v1 of 14 Nov;
  channels_last off; batch 32, bf16, seed 42.
- **What was run:** Run all, once with `PROFILE = 'unified'` (A1, then 20 epochs × 750 steps) and once with
  `PROFILE = 'detection'` (15 epochs × 500 steps). Neither run stopped early; both ran their full length.
- **Criteria** (plan approved 15 Nov; masks are the raw decoder output at 0.5, no class-gating). Epochs are
  counted from 1 here; the JSON files count from 0 (`best_epoch` 19 and 14).

**Unified model** — best epoch 20 of 20 (selection 0.880127):

| # | Criterion | Untrained (CP2) | Measured at the best epoch | Result |
|---|-----------|-----------------|----------------------------|--------|
| S1 | Every loss term and gradient norm finite; best-epoch training loss below epoch 1's | — | 15,000 steps, 0 non-finite values, no `nonfinite.json`; loss 1.704758 → 0.765366 | PASS |
| S2a | Macro one-vs-rest AUC ≥ 0.90; stop if any class < 0.80 | 0.5234 | 0.998221 (real 0.998345, generated 0.999297, edited 0.997022) | PASS |
| S2b | Early stop, or selection gain < 0.005 over the last 4 epochs | — | gain 0.000122 (no early stop) | PASS |
| S3 | (judged on the detection model) GenImage accuracy, 3 classes | — | 0.9838 | REPORT |
| A1 | 64-image overfit: accuracy ≥ 0.99 within 300 steps; per-image mask F1 on the 22 edited ≥ 0.80 within 1,000 steps | — | accuracy 1.0 at step 25; mask F1 0.6145 at step 25, 0.8710 at step 50; 1.91 min | PASS |
| A2 | Real crops: share with any mark ≤ 0.20; mean marked area ≤ 0.01 | 1.0 / 0.1405 | 0.081515 / 0.008319 | PASS |
| A3 | Generated crops: the same limits | — | 0.017868 / 0.004010 | PASS |
| A4 | Validation per-image mask F1 ≥ 0.50 | 0.2162 | 0.762032 | PASS |
| R1 | Called-edited gap, DiffSeg30k real − COCO real; stop if > 0.10 | — | 0.031315 − 0.012422 = 0.018893 | REPORT (no stop) |
| R2 | Validation binary mIoU (guide; not comparable with CP4's P2) | — | 0.661394 | REPORT |

- Other validation numbers at the best epoch: accuracy 0.9753, balanced 0.9657; mask F1 pooled 0.807117,
  IoU pooled 0.676610; gate mean 0.62018 (untrained 0.452).
  - Accuracy per source and class: GenImage real 0.9791 / generated 0.9880; SID-Set real 0.9442 / generated
    0.9848; COCO real 0.9834; DiffSeg30k real 0.9687 / edited 0.9372.
  - Marks on real crops per source (any / area): COCO 0.060041 / 0.006803; GenImage 0.057548 / 0.004727;
    SID-Set 0.131846 / 0.021051; DiffSeg30k 0.179541 / 0.010606.
- Training subset (2,000 images, fixed crops) at the same epoch: selection 0.884048 (validation 0.880127),
  mask F1 0.769206 (0.762032), macro AUC 0.998889 (0.998221): no train–validation gap.

**Detection model** — best epoch 15 of 15 (binary AUC 0.999717):

| # | Criterion | Measured at the best epoch | Result |
|---|-----------|----------------------------|--------|
| S1 | Every loss term and gradient norm finite; best-epoch loss below epoch 1's | 7,500 steps, 0 non-finite values; loss 0.824854 → 0.206964 | PASS |
| S2a | Binary AUC ≥ 0.90 | 0.999717 | PASS |
| S2b | Early stop, or gain < 0.005 over the last 4 epochs | gain 0.000078 (no early stop) | PASS |
| S3 | In-distribution SD v1.4 accuracy ≥ 0.95, by 1 − P(real) ≥ 0.5 | 0.991568 (real 0.9905, generated 0.9925); 0.971673 already at epoch 1 | PASS |

- Gate mean 0.815278; training subset accuracy 0.9955.

- **Compute:**
  - unified session (A1 and training): 3.015 h wall time = **5.85 CU** at 1.94 CU/h
    (`compute.json`: 2.0562 GPU training hours = 3.989 CU);
  - detection session: 1.547 h = **3.00 CU** (`compute.json`: 1.023 h = 1.985 CU);
  - seed 42 in all: **8.85 CU** (plan estimate about 8).
- **Notes for Dhanush:**
  - A2 passed with a small margin: real marked area 0.008319 against 0.01. It swung between epochs during
    the run (0.0060 to 0.0493) and stayed between 0.0060 and 0.0099 from epoch 12, as the learning rate
    decayed. On their own, DiffSeg30k real crops are at 0.010606 and SID-Set real crops at 0.021051.
  - R1: the class decision shows no DiffSeg30k-style shortcut (gap 0.019), but the mask marks DiffSeg30k
    real crops three times as often as COCO real crops (0.180 vs 0.060). Carried to CP4 and BR-Gen.
  - R2 (binary mIoU 0.661 on crops of the training editors) is well below CP4's P2 bar (0.80 on whole images
    of held-out editors). The two are not directly comparable, but localization is the CP4 risk.
  - The gate leans on the semantic tokens more in the detection model (0.815) than in the unified model
    (0.620), where the trace branch carries about 38% of the fused tokens.
  - Both best epochs are the last epochs (the cosine schedule ends at learning rate 0); the plateau test
    passed in both runs, so no extension is needed.
  - The detection JSON's `vs_untrained_cp2` block compares a macro AUC the detection profile does not log
    (`best: null`); CP2 has no untrained detection value. A cosmetic fix for `cp3.py`, carried to the next
    code change.
- **Result:** **PASS** on every criterion for both models (script verdict). Seeds 123 and 2026 wait for CP3
  and CP4 to be accepted.
- **Verified by Dhanush:** DHANUSH

---

### CP4 — Generalization gate, seed 42 — 2026-11-18 — **FAIL** (G1 and G2; RF PASS; no second fix)

- **Judged on:** the frozen development hold-out only. `cp4_dev_split.json`, seed 42, salt `cp4_dev_v1`,
  SHA-256 `1e0792a8f474…`.
  - P1: 20,000 images, 20% of each GenImage validation folder.
  - P2: 486 images (229 on AI-generated bases, 257 on real bases).
  - No other test image was read. Thresholds stay at 0.5.
- **Models:**
  - detection = CP3's `eviground_detection_seed42/best.pt`;
  - unified, gated = the CP4 fix run, `eviground_ctx_unified_seed42/best.pt` (below).
- **Result files** (Drive `Paper 1/runs/checkpoints/`):
  - `cp4_generalization_gate_seed42.json`: P1, G1 and RF. Written on 17 Nov (commit `616bec5`), then
    overwritten on 18 Nov (`33707a3`) with identical P1 values by the mistaken run below.
  - `cp4_generalization_gate_seed42_ctx.json`: G2 (commit `ff1b14d`, 12:53 UTC; P1 reused from the file above,
    `p1_rerun: false`).
  - `localization_diagnosis_seed42.json` and `localization_diagnosis_seed42_ctx.json`: validation only.

**Gate**

| # | Criterion | Measured | Result |
|---|-----------|----------|--------|
| G1 | P1 mean accuracy over 8 generators > 0.721 **and** > the detection CLIP probe, raw and harmonized | detection model 0.7846 raw / 0.7771 harmonized; probe 0.8136 / 0.8015 | **FAIL** (above 0.721, below the probe) |
| RF | Accuracy ≥ 0.98 on every unseen generator (raw) | SD v1.5 0.9969, Wukong 0.9813, Midjourney 0.7342, BigGAN 0.7088, VQDM 0.6758, GLIDE 0.6221, ADM 0.5592 | PASS (not raised) |
| G2 | P2 edited-class IoU (DiffSeg30k mIoU, pooled) ≥ 0.80, unified model | **0.5101** (two-class mIoU 0.5799) | **FAIL** |

**G1, details** (P1 dev, mean over 8 generators; per-generator numbers are in the JSON):

| | detection model | detection probe | unified model | unified probe |
|---|---|---|---|---|
| accuracy, raw | 0.7846 | 0.8136 | 0.8810 | 0.8953 |
| accuracy, harmonized | 0.7771 | 0.8015 | 0.8642 | 0.8777 |
| AP, raw | 0.9479 | 0.9194 | 0.9692 | 0.9563 |

- **The detection model's misses are at the 0.5 threshold, not in its ranking.**
  - Real images are classed correctly 99.4–99.8% of the time.
  - Generated images, raw: ADM 0.122, GLIDE 0.248, VQDM 0.354, BigGAN 0.423, Midjourney 0.473.
  - Its AP is above the probe's (0.948 vs 0.919).
  - Raw and harmonized differ by 0.008, so compression does not explain it.
- **Unified-model rows (reported, not judged):** they are for the CP3 unified model. The fixed model's P1 row
  comes with the full evaluation.
- **Real P1 images called edited** by the CP3 unified model: 3.85%. By tile count: 1: 0.68%; 2–4: 3.05%;
  5–9: 9.07%; 10–16: 15.56%; 17+: 21.64%.

**G2, details** (P2 dev, whole images, 0.5):

| | fixed unified model (gated) | patch probe |
|---|---|---|
| edited IoU, all 486 | **0.5101** | 0.1392 |
| edited IoU, held-out-editor pixels only | 0.4588 | 0.1355 |
| edited IoU, real base (257) | 0.7471 | 0.1574 |
| edited IoU, real base, held-out pixels | 0.6792 | 0.1287 |
| over-marking (share of pixels) / misses (share of edited pixels), all | 0.1339 / 0.3148 | 0.0331 / 0.8489 |
| pixel AUC / mean boundary F1, all | 0.8137 / 0.2320 | 0.6530 / 0.0624 |
| edited images called edited (P(edited) ≥ 0.5) / by the paper's 1% area rule | 0.9753 / 0.9938 | 0.9671 / 0.7016 |

- **The DiffSeg30k paper's SegFormer** reaches 0.951 on its Glide + Kolors hold-out (all images; arXiv
  2511.19111v2, Table 4). Our 0.5101 is far below it, so the localization claim is narrowed.

**The one fix (context crops on AI-generated bases; Dhanush, 17 Nov)**
- **Why:** the localization diagnosis (validation, CP3 unified) found:
  - whole-image edited IoU 0.3946;
  - over-marking 28.9% of pixels;
  - on AI-generated bases, 35.9% of pixels over-marked, IoU 0.3303 (real bases 0.7209).
- **Change:** for 1/3 of the edited training draws on an AI-generated base, a crop is taken anywhere in the image.
  - It is kept at edited share exactly 0, labelled generated with an empty mask.
  - Or kept at 1–90%, labelled edited.
  - `data.context_crop_share = 0.3333`; nothing else changed.
- **Shortcut check (notebook 10, CPU):** no material change.
  - Real vs not real 0.5122 → 0.5120.
  - Every quality-75 and source share moved by ≤ 0.002, apart from the designed move: DiffSeg30k is now
    4.75% of the generated class.
- **Retrain (notebook 11, `runs/eviground_ctx_unified_seed42/`, commit `ff36b71`):**
  - best epoch 15 of 20 (early stop);
  - macro AUC 0.997864;
  - mask F1 0.758295.
  - **A2 FAILED by 0.0003:** real-crop marked area 0.010302 against ≤ 0.01. Dhanush let it go on to
    notebook 08; the bar is unchanged.
- **Whole-image validation (notebook 08), CP3 → fixed:**
  - edited IoU 0.3946 → 0.4335;
  - over-marking 0.2886 → 0.2148 (AI bases 0.3589 → 0.2628);
  - misses 0.1908 → 0.2276;
  - real images called edited 0.0893 → 0.1053.
  - Interior tiles (≥ 90% edited) miss 33.4% of their pixels, against 18.1% elsewhere.
- **The gated model:** Dhanush chose the fixed model for G2 on 17 Nov, from validation only, before any P2
  number existed.

**Run on the wrong model, 18 Nov (recorded, not the gate)**
- Notebook 09 ran with `P2_ENABLED = True` and `RUN_EXPERIMENT` left at `None`, so P2 was scored on the
  **CP3** unified model. Edited IoU 0.4904 (real base 0.7518; held-out pixels 0.4162). It cost 4.33 CU.
- It does not change the gated model.
- A guard now stops a real P2 run unless `RUN_EXPERIMENT == 'eviground_ctx'`. It caught one more wrong
  attempt before any scoring.

- **Compute (from the result files; CU/h as entered in each session):**

| Run | Notebook | Hours | CU |
|---|---|---|---|
| Localization diagnosis, CP3 unified | 08 | 0.827 | 1.60 |
| Sampler shortcut check | 10 (CPU) | 0.082 | not recorded |
| Gate, CP3 models, P2 off | 09 | 1.405 | 2.73 |
| Fix retrain (session; training 3.421) | 11 | 3.056 | 5.20 |
| Localization diagnosis, fixed unified | 08 | 0.652 | 1.26 |
| Gate with P2 on the CP3 model (mistake) | 09 | 2.232 | 4.33 |
| **G2 gate, fixed unified, P2 only** | 09 | 0.370 | **0.72** |
| **CP4 in all** | | | **15.84** |

- **Result:** **FAIL** on G1 and G2. RF passed. The one fix was used (context crops, unified model), and
  there is no second fix. As the CP4 row says, the claims are narrowed:
  - **Detection:** above the GenImage ResNet-50 bar (0.785 vs 0.721) but below a CLIP linear probe trained on
    the same data, in accuracy at threshold 0.5. Above the probe in AP.
  - **Localization:** held-out-editor localization reaches edited IoU 0.510 (0.747 on real bases), not the 0.80
    target. Interior misses and over-marking on AI-generated bases are the stated limits.
  - The **final configuration** is the context-crop unified model, which seeds 123 and 2026 use.
- **Verified by Dhanush:** ☐

### CP3 training health of the extra seeds (final configuration) — 2026-11-18

Notebook 13 (L4, commit `911e680`) trained the unified model in the final configuration (context crops,
`data.context_crop_share = 0.3333`) with seeds 123 and 2026, and ran the CP3 checks on each. The A1 overfit record of CP3
was reused (a pipeline check, not a seed one). Values are on validation crops, at each run's best epoch.

| Seed | Best epoch | Selection | Macro AUC | Pooled edited IoU | Mask F1 (A4 ≥ 0.5) | Real crops: any mark (A2 ≤ 0.2) / area (A2 ≤ 0.01) | Generated crops: any / area (A3) | Verdict |
|---|---|---|---|---|---|---|---|---|
| 42 (17 Nov, notebook 11) | 15 | 0.87808 | 0.997864 | 0.668879 | 0.758295 | 0.10532 / **0.010302** | 0.0345 / 0.0067 | FAIL on A2 only |
| 123 | 18 of 20 | 0.87056 | 0.997629 | 0.650245 | 0.743497 | 0.102074 / 0.008986 | 0.029512 / 0.004327 | **PASS** |
| 2026 | 12 (early stop at 17) | 0.87596 | 0.997456 | 0.666702 | 0.754473 | 0.102435 / **0.010192** | 0.041558 / 0.005523 | FAIL on A2 only |

- **Over the three seeds:**
  - selection 0.8749 ± 0.0039;
  - pooled edited IoU 0.6619 ± 0.0101;
  - macro AUC 0.99765 ± 0.00020 (mean ± sample sd).
- **Reported checks (seeds 123 / 2026):**
  - S3 GenImage 3-class accuracy 0.9848 / 0.9826;
  - R1 gap in "called edited" 0.0335 / 0.0356;
  - R2 binary mIoU 0.6385 / 0.6540.
- **A2:** two of the three seeds exceed the 0.01 area bar by 0.0002–0.0003. That is a property of the configuration, not of
  one seed. **Decision (Dhanush, 18 Nov):** each is recorded as FAIL, the bar is unchanged, and all three seeds are kept
  for the three-seed results. A2 is reported per seed in the training-health appendix.
- **Compute:** the session ran 5.77 h on an L4. Dhanush's Colab balance fell from about 44.3 to about 31–32 CU, so
  about 12.5–13 CU. The notebook's own record (1.7 CU/h) says 9.805; the real rate is about 2.2 CU/h.
- **Verified by Dhanush:** ☐

### CP5 — Evidence faithfulness, seed 42 — 2026-11-19 — **FAIL** (fallback applied; the deletion test is repaired, notebook 15)

- **Run:** notebook 14 (L4, commit `1973329`), on the unified final configuration, seed 42.
  - The go/no-go uses 200 edited validation images (training editors) and 30 randomization tiles.
  - P6 (1,500 images) is reported only.
- **Rule** (Dhanush, 19 Nov):
  - EPG ≥ 1.5 × the mean area fraction with a one-sided paired Wilcoxon p < 0.01;
  - last-step randomization SSIM < 0.5;
  - deletion AUC below random-order deletion.

| Map | EPG ratio (p) | SSIM, last step | Deletion vs random | Result |
|---|---|---|---|---|
| Grad-CAM++ | 1.38× (2.3e-7) | 0.455 | 0.919 vs 0.832 | fail (EPG, deletion) |
| Mask decoder | 2.00× (4.1e-33) | 0.545 | 0.921 vs 0.832 | fail (SSIM, deletion) |

- **The pixel deletion test was confounded:** the blur that removes pixels is itself a local edit, so P(edited) stayed near 0.9 in every order (deletion ≈ insertion).
- **Decision (Dhanush, 24 Nov):** CP5 is FAIL. The spec's fallback applies: **the mask decoder is the primary evidence map**, Grad-CAM++ secondary, and CP6 uses the decoder map.
- **Repair of the test:** one pre-registered repair (spec cards `sg_jrepjxk8bs`, `sg_rdyd3wa8x3`): token deletion before fusion, oracle validity first, and Spearman beside SSIM, in notebook 15. Its result is added here.
- **Compute:** about 7 CU by balance (28.19 left).
- **Verified by Dhanush:** ☐
