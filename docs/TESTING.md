# Testing log

Commands and results for every milestone, newest first. The local checks (lint, tests, notebook dry
runs) run on CPU with synthetic data: no GPU, no Kaggle or Hugging Face access, no real images.
Entries titled as a notebook "run" record real Colab sessions (GPU or CPU runtime) instead; their numbers
are read from the result files the notebooks write to Drive.

## PR X — Drive test-set tars and the P5 freeze (notebook 16, CPU) — 2026-11-24

- **`data/testcache.py`:** one tar per test set, holding the **original file bytes** (images and masks, not re-encoded) and a manifest. The manifest records the SHA-256 of each member and of the tar, and each image's row with the paths left out.
  - **`build`:** writes the tar and the manifest.
  - **`publish`:** copies both to Drive, re-hashes the tar **from Drive**, and decodes one seeded random member of the Drive copy (`drive_sha256_ok`, `decode_check`). If either check fails, nothing is written to Drive.
  - **`is_ready`:** true when a verified tar of the recorded size is on Drive.
  - **`stage`:** copies the tar and checks it and every member against the manifest. It refuses unknown or missing members, unpacks, and returns rows that `testsets.file_items` reads unchanged.
- **`subsets.p5_subset`:** P5 at about half the spec's size (Dhanush, 24 Nov): 250 real + 250 generated per GenImage generator, drawn from the frozen P1 subset, plus 500 DiffSeg30k held-out images from P2. The images are already in the P1 and P2 tars, and the perturbations are applied at run time.
- **`cp5.aspect_sensitivity`:** CP5 per part, as run and without the 26 aspect-mismatched SID-Set masks: the EPG ratio per map (notebook 14) and the token-deletion verdict (notebook 15). This is a sensitivity check only.
- **Notebook 16** (`make_nb16.py`, CPU runtime, about 0.2–0.3 CU estimated):
  - freezes P5 into `eval_subsets.json`, or reuses it if already frozen;
  - builds and publishes `p1_subset`, `p2`, `p3_brgen`, `p3b`, `p4` and `claims_edit_src`, one at a time, freeing the staged files after each; a set already verified on Drive is skipped;
  - prints the CP5 sensitivity table if the notebook 14/15 files are on Drive;
  - does one full staging round trip (`p2`), then writes `test_cache_record.json` and `compute_test_cache.json`.
- **Checks (sandbox):** ruff clean; pytest 508 passed; repo audit ok; notebook 16 dry run passed (fake world, every set, Drive re-hash and decode check). CI step added.
- **Not run here:** the real data, sizes and timing on Colab.

## PR W — compute record per GPU type; SID-Set aspect-ratio exclusion pre-registered — 2026-11-24

- **Compute record (`utils/session.py`):**
  - `CU_RATES` gives the rate per runtime type: L4 2.2 and T4 1.28 CU/h, both measured from Dhanush's balance (notebook 13; notebook 15: 1.71 CU over 1.3405 h), plus CPU 0.08.
  - `finish(..., gpu=...)` takes the rate of the GPU the session ran on and writes `gpu_type`, `cu_estimate`, `cu_balance_start`, `cu_balance_end`, `cu_measured` and `cu`.
  - `cu` is the balance drop when both balances are known (`cu_source: balance`), otherwise the rate estimate. `compute_units` stays as an alias.
  - `with_balance(info, balance_end)` returns the record with the balance read after the run (no file I/O).
  - Notebooks 14 and 15 pass the GPU name.
  - **Why:** notebook 15 ran on a T4 and its record priced it at the L4 rate (2.949 CU against 1.71 by balance).
- **SID-Set aspect-ratio exclusion (`eval/p6data.py`; spec card `sg_nj9cpnxxvd`, decided after CP5, prospective):**
  - A SID-Set mask whose aspect ratio differs from its image's by more than 2% is resized (the notebook 14 rule) and flagged `mask_aspect_mismatch`; it no longer stops the run.
  - Any other part still stops on such a mask (none was found).
  - `mask_metric_items` is the one filter for EPG, HR-GT and CC: flagged masks are left out unless `include_mismatched=True` (the sensitivity check).
  - Items from the notebook 15 P6 tar, cached before the flag existed, are recognised from their recorded mask size (`is_aspect_mismatched`).
  - **Review fixes (PR #54):**
    - SonarCloud S2083 (blocker): the balance helper that read and rewrote a JSON file is replaced by the pure `with_balance(info, balance_end)`.
    - Codex P1: notebook 14 writes `mask_aspect_mismatch` into each P6 record. `faithfulness_by_part` stays the as-run CP5 record (all masks); `faithfulness_by_part_excl_mismatched` is the sensitivity check (`p6data.record_variants`). Records resumed from before the flag count as not flagged, and their number is saved.
    - The rule applies prospectively (CP6 HR-GT, Phase C CC, later EPG); the CP5 as-run numbers are unchanged (Dhanush, 24 Nov).
  - `resize_summary` counts the flagged masks per part.
- **Checks (sandbox):**
  - ruff clean; `pytest` 497 passed; `repo_audit --history` ok; notebook 14 and 15 dry runs passed.
  - New tests: rate per GPU, balance preferred, `with_balance`, unknown GPU; SID-Set flagged not raised, DiffSeg still raises, the filter's default and sensitivity modes, old cache entries recognised.

## Notebook 15 run: CP5 token deletion and randomization rerun (Colab, **T4**) — 2026-11-24

Run on main at `1050c24`, files written 13:18 UTC: `cp5_token_deletion_seed42.json`, `p6_token_deletion_seed42.csv`, `p6_token_deletion_records_seed42.jsonl`, `cp5_token_curves.png`, `compute_cp5_token.json`, and the P6 cache `runs/cache/p6_images.tar` (1,500 images, SHA-256 `1ae27a38fce2…`) with its manifest.

- **GPU:** the record says `Tesla T4`; the run was planned for the L4. Autocast was bf16 as in the L4 notebooks. The notebook 14 maps it reads came from the L4.
- **Cell 4 stopped once** on the 2% aspect-ratio guard (a 764×1024 mask on a 1024×1024 SID-Set image). With Dhanush's approval a Colab cell resized every mismatched mask to its image (the notebook 14 rule) and listed the aspect mismatches, then built the cache without staging again. Resized: SID-Set tampered 26 of 400 (all aspect mismatches, from sizes such as 764×1024, 1024×683 and 527×1024), DiffSeg30k 0, BR-Gen 0. The list is in the result file under `p6_mask_resize`.
- **Token deletion (300 `P6_deletion` images, 10 random orders, 12.26 min):**

| | All | DiffSeg30k held-out | BR-Gen | SID-Set tampered |
|---|---|---|---|---|
| Oracle vs random (p) | 0.511 vs 0.569 (5.3e-6), **valid** | 0.692 vs 0.823 (2.6e-7), valid | 0.510 vs 0.544 (0.008), valid | 0.331 vs 0.340 (0.52), **not valid** |
| Decoder | **0.435, PASS** (p 2.5e-35) | 0.630, pass | 0.438, pass | 0.238 (uninformative) |
| Grad-CAM++ | **0.503, PASS** (p 1.7e-17) | 0.704, pass | 0.494, pass | 0.310 (uninformative) |

  - Without the 9 deletion images whose SID-Set mask was stretched: oracle 0.512 vs 0.571 (p 6.5e-6), decoder 0.436, Grad-CAM++ 0.504, both pass; SID-Set still not valid.
- **Randomization, SSIM (Spearman) per cumulative step:**
  - validation Grad-CAM++: heads 0.287 (0.113), fusion 0.287 (0.091), trace 0.243 (0.040), LoRA 0.376 (0.043);
  - validation decoder: 0.123 (−0.045), 0.122 (−0.044), 0.546 (0.184), **0.546 (0.130)**;
  - P6 Grad-CAM++: 0.325 (0.129), 0.271 (0.065), 0.117 (0.043), 0.260 (0.011);
  - P6 decoder: 0.085 (0.017), 0.085 (0.017), 0.289 (0.027), 0.289 (0.002).
  - Grad-CAM++ on validation ended at 0.376 here and 0.455 in notebook 14 (same tiles); the randomized weights differ between runs.
- **Verdict:** CP5 stays FAIL (decoder fails randomization SSIM, Grad-CAM++ fails EPG); decoder primary. Recorded in `docs/CHECKPOINTS.md`.
- **Compute:** 1.3405 h wall. **1.71 CU by balance** (28.19 → 26.48). The record's 2.949 CU uses the L4 rate (2.2 CU/h), which does not apply to a T4; its `cu_balance_start` reads 28.02.

## Notebook 14 run: CP5 on validation and the P6 explanation pass (Colab L4) — 2026-11-19

Run on main at `1973329`; the result files were written at 18:56 UTC. Results are read from Drive: `cp5_evidence_faithfulness_seed42.json`, `compute_cp5.json`, `p6_randomization_seed42.json`, `p6_faithfulness_seed42.csv` and the bundles in `runs/explain/p6_unified_ctx_seed42/`.

**CP5 on validation: FAIL** under the agreed rule (200 edited validation images; randomization on 30 best tiles):

| Map | EPG / area (bar 1.5×, p < 0.01) | Randomization, last-step SSIM (bar < 0.5) | Deletion AUC vs random (must be lower) | Insertion AUC vs random |
|---|---|---|---|---|
| Grad-CAM++ | 0.4060 / 0.2952 = **1.38×** (p 2.3e-7): fail | **0.455**: pass | 0.919 vs 0.832: fail | 0.922 vs 0.832 |
| Mask decoder | 0.5910 / 0.2952 = **2.00×** (p 4.1e-33): pass | **0.545**: fail | 0.921 vs 0.832: fail | 0.936 vs 0.832 |

- **The deletion test was confounded.** Removed pixels are replaced by a blurred copy, and a blurred patch inside a sharp image is itself a local edit. So P(edited) stayed near 0.9 in every order, and deletion and insertion AUCs came out almost equal (0.919 / 0.922).
- **The decoder's randomization SSIM** falls to 0.21 once the decoder is randomized, then rises to 0.54 when the trace branch is randomized too. That is SSIM between two near-flat maps.
- **Decision (Dhanush, 24 Nov):**
  - CP5 recorded as FAIL;
  - the spec's fallback applied (the mask decoder is primary, Grad-CAM++ secondary; CP6 uses the decoder map);
  - one pre-registered repair of the test (PR U, notebook 15).
- **P6 (1,500 images, all written as bundles):**
  - **EPG ratio, Grad-CAM++ / decoder:** all 1.16× / 1.65×; DiffSeg30k held-out 1.29× / 1.78×; SID-Set tampered 1.42× / 3.04×; BR-Gen 1.00× (p 0.52) / 1.20×.
  - **Randomization last step (50 tiles):** Grad-CAM++ 0.507, decoder 0.290.
  - **Deletion (300):** 0.890 / 0.912 vs random 0.730; insertion 0.878 / 0.919 vs 0.732.
- **P6 masks of another size:** some real P6 masks are not the size of their image, and cell 7 crashed on them. Dhanush patched cell 7 in Colab: such a mask is resized to the image with `F.interpolate(mode='nearest')`, then > 0. The P6 numbers here were computed with that patch. The loaders now do the same (PR U), and notebook 15 reports how many masks were resized, per part.
- **P6 verdicts:**
  - edited images → 1,040 edited, 150 real, 10 generated;
  - generated → 79 generated, 45 edited, 26 real;
  - real → 108 real, 31 edited, 11 generated.
- **Compute:**
  - the record covers only the last session (0.639 h, 1.405 CU at 2.2 CU/h; staging 9.28 min, the P6 pass 5.1 min in that session);
  - Dhanush's balance after the run: **28.19 CU**, about **7 CU** spent against an estimate of 3.0. The difference was an earlier, slower or restarted session that the notebook did not record.

## PR U — CP5 repair: token-level deletion before fusion, the P6 cache (notebook 15) — 2026-11-24

- **Rules pre-registered in the spec** (cards `sg_jrepjxk8bs`, `sg_rdyd3wa8x3`, waiting for Dhanush) before notebook 15 runs:
  - **Removal:** fused-grid positions in attribution order; at each removed position the semantic patch token **and** the trace token become their per-tile mean token; fusion + gate + class head are re-run. The CLIP class token is kept.
  - **Orders:** oracle, decoder, Grad-CAM++, and the mean of 10 seeded random orders.
  - **Validity first:** the oracle must beat random with p < 0.01, else the test is uninformative.
  - **Per-map pass:** below random with p < 0.01.
  - **Randomization:** SSIM < 0.5 stays the criterion; Spearman is reported beside it.
- **`EviGround.branch_tokens` / `logits_from_tokens`:** the branch tokens before fusion, then fusion → class head. `forward` now shares `_trace_tokens`.
- **`explain/cp5.py`:** `token_scores`, `token_deletion`, `token_deletion_verdict`, `bundle_tile`, `spearman`. `randomization` also reports Spearman per step.
- **`eval/p6data.py`:** `write_cache` / `read_cache` / `cached_items`, a tar of the decoded P6 images and masks with a SHA-256 manifest. Any changed member stops the run.
- **`p6data.align_mask`:** the notebook 14 cell 7 patch moved into the loaders. An edit mask of another size is resized to its image with `torch.nn.functional.interpolate(mode='nearest')`, then > 0, for all three edited parts (DiffSeg30k's editor map too). The original size goes to `meta['mask_resized_from']`, and the cache stores the aligned mask. A mask whose aspect ratio differs from its image's by more than 2% stops the run (Dhanush, 24 Nov). `resize_summary` gives the count per part, which notebook 15 prints and saves as `p6_mask_resize`.
  - **My miss:** the loaders never compared mask and image sizes, and the fake world had only same-size masks, so no dry run could catch it. The dry world now has half-size SID-Set and DiffSeg30k masks.
- **`session.finish(..., balance_start=)`:** records `cu_balance_start`.
- **Notebook 15** (`make_nb15.py`, L4, estimate 1.5 CU):
  - `CU_BALANCE_START` is required;
  - the P6 cache is reused if present, else built from staging;
  - token deletion on the 300 `P6_deletion` images, with resume;
  - randomization reruns on the 30 validation and 50 P6 tiles;
  - result file, CSV, curves and compute record, then `release()`.
  - A CI step runs its dry run.

### Checks (sandbox, CPU)
- **New tests:**
  - `test_token_hooks_reproduce_the_forward_logits`;
  - token deletion with a synthetic head: the oracle beats random, a good map passes, a random map fails, and a flat head is flagged uninformative;
  - `token_scores` and Spearman;
  - `bundle_tile`;
  - the cache round trip, and a changed member refused;
  - `align_mask`: unchanged at the same size, equal to the notebook 14 rule on a half-size mask, a 1% rounding difference accepted, another aspect ratio refused; each edited part aligned from a half-size mask (DiffSeg30k editor values kept), the loader stopping on a misshaped mask, and the cache keeping the aligned mask.
- **Notebook 15 dry run:** passed.
  - **One-off check** (`/tmp/claude-0/-home-user-Paper1/d56e2aef-4c2a-5b25-a13a-a24272263c08/scratchpad/nb15chk/run.py`): running the cache and token cells a second time reported "P6 cache reused" and "3 images done earlier, 0 to go".
  - The dry verdict was UNINFORMATIVE, as expected for an untrained tiny model.
- Notebook 14 regenerated unchanged before the mask alignment; it now carries the half-size dry-run masks.
- **After the mask alignment (24 Nov):** `ruff` clean, `pytest` 488 passed, `repo_audit --history` ok; the dry runs of notebooks 14 and 15 passed, with half-size SID-Set and DiffSeg30k masks resized in both.
- **Not run here:** the real model, the real data, and L4 timing.

## PR T — notebook 14: CP5 evidence faithfulness and the P6 explanation pass (code, CPU-tested) — 2026-11-19

The code and notebook for CP5. Nothing has run on real data yet: notebook 14 runs on an L4 (Dhanush, 19 Nov 2026: L4 only, one session).

- **CP5 rule** (spec §9.1; decided by Dhanush on 19 Nov), checked for both maps, Grad-CAM++ and the mask decoder:
  - **EPG:** mean EPG ≥ 1.5 × the mean area fraction of the edit, and a one-sided paired Wilcoxon test with p < 0.01;
  - **randomization:** mean SSIM after the last cumulative step (heads → fusion → trace → LoRA) < 0.5;
  - **deletion:** mean deletion AUC below the random-order AUC.
  - **Result:** Grad-CAM++ passes all three → `PASS` (Grad-CAM++ is primary). Otherwise the decoder is primary → `FALLBACK` (the spec's fix). Neither passes → `FAIL`, stop and ask.
  - The go/no-go uses validation images only: 200 edited (training editors) with seeded salt `cp5_val_v1`, 30 tiles for randomization, and 20 overlays.
- **New `explain/cp5.py`:**
  - `explain_image`: probabilities, gate, decoder map, Grad-CAM++ for the edited class and for the predicted class;
  - `faithfulness_record`: EPG, area fraction, and deletion / insertion against a random order on the best tile;
  - `randomization`: one model copy per batch of tiles, both maps;
  - `map_verdict` / `cp5_verdict`, `summarize`, `overlay_figure`.
- **New `eval/p6data.py`:** P6 images by part:
  - DiffSeg30k masks by the training rule;
  - SID-Set and BR-Gen masks binarized at > 127;
  - Chameleon read from the zip;
  - BR-Gen files checked against notebook 12b's SHA-256 (`check_brgen`).
- **`EviGround.predict_tiled`:** now also returns the mean gate (`None` when a forward has no gate).
- **Notebook 14** (generator `make_nb14.py`; L4):
  - pre-flight at 2.2 CU/h, estimate 3.0 CU;
  - stages the shards and the P6 files;
  - runs CP5 on validation;
  - writes a P6 evidence bundle per image, plus faithfulness (EPG on all edited images, deletion on `P6_deletion`). The pass resumes from `p6_records.jsonl`.
  - Randomization on 50 P6 tiles.
  - Writes `cp5_evidence_faithfulness_seed42.json`, `p6_faithfulness_seed42.csv`, `cp5_overlays_val.png` and `compute_cp5.json`, then releases the runtime.
  - A CI step runs its dry run.

### Checks (sandbox, Python 3.13.18, CPU)
- `tests/test_explain.py`: 17 passed (3 new: the verdict rules, including the 1.25× ratio, too few images for p < 0.01, unchanged maps and deletion no better than random; summaries and the overlay file; the CP5 functions on the tiny model).
- `tests/test_p6data.py`: 3 passed (every part with its mask rule; the BR-Gen hash check and missing files; the manifest rows).
- **A first full run failed twice** in `test_models.py`: the fake forwards in those tests return no `gate`. Fixed by making the gate optional in `predict_tiled`.
- **`EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/14_cp5_explanations.ipynb`:** passed (fake world, tiny model trained one epoch, fake P6 of 9 images, fake BR-Gen and Chameleon zips).
  - The dry verdict was `FAIL`, as expected for an untrained tiny model.
  - **One-off resume check** (`/tmp/claude-0/-home-user-Paper1/d56e2aef-4c2a-5b25-a13a-a24272263c08/scratchpad/nb14chk/run.py`, not committed): the P6 cell run a second time reported "9 images done earlier, 0 to go".
- **The fake world has no edited validation image in its shards,** so the dry run (only) uses edited training rows, as notebook 08's does. The real run requires edited validation images.
- Dry runs of notebooks 08, 09, 12, 12b and 13: passed.
- `ruff check src tests`: clean. `python -m pytest -p no:warnings`: 476 passed. `python -m eviground.utils.repo_audit --history`: ok.
- **Not run here:** the real model, the real data and L4 timing. The 3.0 CU figure is an estimate.

## Notebook 12b run: frozen P6 subsets (Colab CPU) — 2026-11-18

Run on main at `a843a74`, finished 23:29 UTC; wall 0.182 h, **0.015 CU** at 0.08 CU/h (estimate 0.05). Results are read from Drive:
`p6_subsets_record.json`, `compute_p6_subsets.json` and the new keys of `eval_subsets.json` (P1 and P4 untouched).

| Subset | Images | Per part (DiffSeg30k held-out / SID-Set tampered / BR-Gen / generated / real) | SHA-256 |
|---|---|---|---|
| P6 | 1,500 | 400 / 400 / 400 / 150 / 150 | `c12ba888637b8f6fce0f4ac9f67d180823062f31d39c01ad7736b4a55df17415` |
| P6_mllm | 500 | 134 / 133 / 133 / 50 / 50 | `1aa9bb7e9096e0b8572e341f382addab98fb40c9099715c8b069c0005442976e` |
| P6_deletion | 300 | 100 / 100 / 100 / – / – | `c42ed65cea118494de2f89319dc8b2a1f617f9d47a4d16f643dedf908b267ab3` |

- **Pools:**
  - DiffSeg30k held-out 1,948;
  - SID-Set tampered / real / generated 2,000 each;
  - P1 generated 40,000;
  - GenImage real outside the dev split and P1 30,000;
  - training pool 121,229 rows.
- **The 150 real images are GenImage 50, SID-Set validation real 50 and Chameleon 50.**
  - The manifest has **no unedited DiffSeg30k validation image**: all 3,192 `unused_validation` rows carry editors, so that pool is empty.
  - The approved fallback filled the 50 from SID-Set validation real (`sidset_top_up: 50`).
  - The record still lists these 50 under the frozen key `diffseg_unedited`, but they are SID-Set images.
  - These SID-Set real images are `test_P3b` rows, so they also appear in the P3b evaluation (a different protocol).
- **Training check:** 0 real images skipped as exact or near (pHash ≤ 6) duplicates of training, in every source.
- **Chameleon:** 26,033 images unpacked to the local disk; 12,363 real outside P4; the first 400 in P6 rank order hashed in 116 s.
- **BR-Gen:**
  - the 400 chosen images and their masks were extracted with 7z, fully decoded and size-checked in 517 s, with 0 problems;
  - file-list hash `e58434f264fca68d79db88aa1db24c03e6297a205933f2a8eba96223cc4edf58` (per-file SHA-256 in `P6_brgen_files`).
- **The runtime was released** by the notebook's last cell.

## Notebook 13 run: unified seeds 123 and 2026 (Colab L4) — 2026-11-18

Run on main at `911e680`, 17:05–22:50 UTC; wall 5.768 h. Results are read from Drive: `training_health_unified_ctx_seed{123,2026}.json`,
`compute_unified_seeds.json`, and each run's `summary.json` / `compute.json`.
- **Seed 123:**
  - 20 epochs, best at 18, selection 0.87056;
  - health **PASS**: macro AUC 0.997629, mask F1 0.743497, real-crop area 0.008986;
  - 181.1 min of training.
- **Seed 2026:**
  - early stop at epoch 17, best at 12, selection 0.87596;
  - health **FAIL on A2 only**: real-crop area 0.010192 against the 0.01 bar; macro AUC 0.997456, mask F1 0.754473;
  - 153.8 min of training.
  - Kept, as seed 42 was (Dhanush, 18 Nov).
- **No NaN or inf** in either run (S1); both stayed within the same configuration (index `1085a527…`, CLIP `32bd6428…`).
- **Compute:**
  - the notebook recorded 9.805 CU at 1.7 CU/h;
  - Dhanush's balance fell from about 44.3 to about 31–32 CU (about 12.5–13 CU), so the real L4 rate is about 2.2 CU/h;
  - the CU in earlier rows recorded at 1.7 or 1.94 CU/h are therefore low by roughly 15–30%.
- **The runtime was released** by the notebook's last cell.

## PR Q — CP5 explanation code and the frozen P6 subsets, notebook 12b (CPU) — 2026-11-18

Code and tests only, on CPU with the tiny model and hand-made maps; no real image, no real model and no GPU in the sandbox. The CP5 run itself is notebook 14 (GPU, later, after its own estimate and Dhanush's go-ahead). Design as approved by Dhanush on 18 Nov 2026.

**New `src/eviground/explain/`:**
- **`gradcam.py`:** Grad-CAM++ on the fused 16×16 tokens Z (`gradcam_pp` returns the raw relu map, upsampled to the tile). `tiled_map` stitches the raw tile maps like the mask and then normalizes once per image by the image max (no per-tile normalization). `decoder_map` (the CP5 fallback) and `alpha_map` (ablation 4).
- **`regions.py`:** connected components with `scipy.ndimage.label` (4-connectivity, components under 0.5% dropped); box, area %, mask confidence, attribution mass, frequency anomaly (8×8 block-DCT high-frequency energy, u+v ≥ 8) and noise anomaly (variance of the image minus its 3×3 mean); `top_attribution_regions` (A1–A3); `rle_encode` / `rle_decode`.
- **`evidence.py`:** the §8 M5 record (E regions for an edited verdict, A1–A3 for fully generated, none for real; G1 branch reliance, G2/G3 image anomalies; unique IDs). The region pixel masks go to a side file keyed by region ID; the LLM record carries no mask. `save_bundle` / `load_bundle`: the evidence JSON, the region RLEs, float16 maps at 1/4 resolution, the probabilities and the gate (what CP6 reads).
- **`faithfulness.py`:** EPG and the area fraction; one `remove_region` (blur baseline) shared by deletion/insertion and counterfactual consistency; RISE deletion/insertion AUC with a random-order baseline; `best_tile`; Gaussian SSIM; cascading randomization (heads → fusion → trace → semantic LoRA) on a deep copy.

**P6 (`eval/subsets.py`):** `P6_PARTS` (400 DiffSeg30k held-out, 400 SID-Set tampered, 400 BR-Gen without SDXL, 150 real, 150 generated); `TrainingIndex` (exact SHA-256 / pixel SHA-256 and pHash ≤ 6 against the training pool); `take_clean`; `p6_pools`; `even_over`; `chameleon_candidates`; `p6_subset`, which builds P6, `P6_mllm` (500, by part) and `P6_deletion` (300, edited only) and stops if any part is short.

**Notebook 12b `12b_p6_subsets.ipynb`** (generator `make_nb12b.py`; CPU; estimate 0.05 CU):
- pre-flight; checks that P1 and P4 in `eval_subsets.json` and `brgen_subset.json` are intact;
- unpacks Chameleon again to the local disk and hashes the first 400 real images outside P4 in P6 rank order (the pick equals the one from hashing all of them while 50 pass; tested);
- builds P6, extracts the 400 BR-Gen images and their masks with 7z, opens them, checks image and mask sizes, and records their SHA-256 (`P6_brgen_files`) for notebook 14;
- adds `P6`, `P6_mllm`, `P6_deletion` and `P6_brgen_files` to `eval_subsets.json`, leaving P1 and P4 untouched; an existing P6 is kept only if identical, otherwise it stops;
- writes `p6_subsets_record.json` and `compute_p6_subsets.json`, then releases the runtime.
- **Dependency:** `scipy>=1.11` (preinstalled on Colab). **CI:** a step for the notebook 12b dry run.

### Checks (sandbox, Python 3.13.18, CPU)
- `tests/test_explain.py`: 12 passed (components, RLE round trip, noise anomaly, EPG, `remove_region`, deletion below random order, `best_tile`/SSIM, the evidence record by verdict, the bundle round trip, Grad-CAM++ on the tiny model, a blank tile next to a strong one stays below 0.01 after stitching, randomization changes the map).
- **A bug found by the randomization test:** the first `_randomize` zeroed the LayerNorm weights, so the randomized map was all zero. It now calls each layer's `reset_parameters()`; SSIM after each step is 0.05–0.10 on the tiny model.
- `tests/test_eval_subsets.py`: 11 passed (4 new: the P6 mix, real-source split and SID-Set top-up, nested subsets, deletion edited only, deterministic hash; exact and near duplicates refused; a short part stops; hashing only the Chameleon head gives the same pick). One fixture was wrong at first (its training pHash sat one bit from the test rows'), and the code refused those rows correctly.
- **`EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/12b_p6_subsets.ipynb`:** passed. Fake world: P6 1,500 (400/400/400/150/150); real 50/50/50 with 20 topped up from SID-Set; `P6_mllm` 134/133/133/50/50; `P6_deletion` 100/100/100; 400 BR-Gen files extracted with 7z, 0 problems.
- Dry runs of notebooks 12 and 13: passed.
- `ruff check src tests`: clean. `python -m pytest -p no:warnings`: 468 passed (470 after the review fixes). `python -m eviground.utils.repo_audit --history`: ok.
- **Not run here:** the real data (the manifest, Chameleon, BR-Gen) and anything on the real models; Grad-CAM++ speed on an L4.
- **Review fixes (SonarCloud and Codex on `08d6cff`):**
  - S3776 (critical): `p6_pools` classifies each row in a helper, `_p6_pool`; the rules are unchanged. S9073: the composite test assertions are split.
  - Codex P1: the Grad-CAM++ denominator used the sum over positions of A·g³. It is now 2g² + (sum over positions of A)·g³, in `channel_weights`, tested against a loop written from the formula.
  - Codex P2: when fewer than 10% of pixels carry attribution, the 90th-percentile cutoff is 0, and the A regions took in the zero background. Only positive pixels are kept now (test: a 100-pixel square gives one 100-pixel region).
  - Codex P2: notebook 12b decodes each BR-Gen image and mask (`load()`), not only the header, before recording its hash.
  - After the fixes: `tests/test_explain.py` 14 passed; full suite below.

## PR P — notebook 13: the unified model, seeds 123 and 2026, in one L4 session — 2026-11-18

**Notebook 12 ran first** (Colab CPU, 0.0977 h, **0.008 CU**). Results from its printout and `eval_subsets_record.json`:
- **`Chameleon.zip`:**
  - 2,683,359,341 bytes; 26,033 members, all images;
  - real 14,863, generated 11,170;
  - unpacked to the local disk, 0 unreadable, 0 unlabelled;
  - copy 60.6 s, unpack 47.6 s, open 135.6 s.
- **P4:** 2,500 real + 2,500 generated (`balanced`). SHA-256 `50167dd2d6df9dc46ade467822f27449c9dcef46bd316810bffb7fe67b5a2cb3`.
- **P1:** 20,000 images, 1,250 per generator and label in all 16 strata (ADM, for example, has 4,800 per label available outside the dev split). SHA-256 `881098bb04db866dc879478551f86c7bc57fcd643cad2bfb06d54c55b93d6dad`.
- The runtime was released by the notebook.

**Notebook 13 `13_unified_seeds.ipynb`** (generator, built from notebook 11's), for Dhanush's item 4: unified seeds 123 and 2026 first, both in one session.
- **Configuration:** the final configuration (`data.context_crop_share = 0.3333`, experiment `eviground_ctx`), CP3's schedule, CP3's A1 record reused (A1 checks the pipeline, not the seed).
- **Pre-flight:** an estimate of 7.5 CU. It requires the A1 record and seed 42's run of the final configuration.
- **Shards:** staged once.
- **Seeds:** one after the other. A seed whose `summary.json` exists is skipped on a rerun; a seed in progress resumes (the trainer checks the seed and the context share).
- **Per seed:**
  - curves;
  - the CP3 checks;
  - a comparison with seed 42 on validation crops (`fixcompare.compare_runs`, each run at its best epoch);
  - the result file `training_health_unified_ctx_seed{seed}.json`.
- **Last cell:** writes `compute_unified_seeds.json` (with each run's `compute.json` CU), then releases the runtime.
- **CI:** a step for the notebook 13 dry run.

### Checks (sandbox, Python 3.13.18, CPU)
- **`EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/13_unified_seeds.ipynb`:** passed (fake world, tiny model, both seeds trained, both result files written, release dry).
- **One-off rerun check** (`/tmp/claude-0/-home-user-Paper1/d56e2aef-4c2a-5b25-a13a-a24272263c08/scratchpad/nb13chk/run.py`, not committed): cells 5 and 7 run again in the same kernel, and both seeds are reported as "finished earlier, skipped".
- `ruff check src tests`: clean. `ruff format --check src tests`: 93 files already formatted.
- `python -m pytest -p no:warnings`: 452 passed. `python -m eviground.utils.repo_audit --history`: ok.

### Review fixes (PR #47)
- **SonarCloud S1192 (critical):** the `'summary.json'` literal was repeated. It is now the constant `SUMMARY`.
- **SonarCloud S3776 (critical, complexity 21 against 15) and S117 (minor):** `plot_curves(RUN)` is split into `_loss_axis`, `_auc_axis`, `_mask_axis` and `_marks_axis`, with the parameter renamed `run`.
- **Rerun:** the notebook 13 dry run passed.

### Not run here
- The L4 session itself: about 7.5 CU (estimated), only after Dhanush's go-ahead.

## PR O — frozen evaluation subsets: Chameleon (P4) and P1, notebook 12 (CPU) — 2026-11-18

**Why (Dhanush, 18 Nov):**
- **Chameleon was incomplete:** `05_Chameleon` held only 1,412 loose images (1,280 real, 132 fake, from the 13 Nov data probe). The authors' full `Chameleon.zip` (2.68 GB) is now a Drive shortcut in that folder.
- **P4:** a seeded, class-balanced 5,000 subset (2,500 + 2,500, or stratified if a class is short), frozen by SHA-256 before any scoring. Seed 42 only.
- **P1:** 1,250 real + 1,250 generated per generator from the 80% outside the CP4 dev split (decided the same day).

**New code:**
- **`src/eviground/data/chameleon.py`:**
  - `label_of`: `0_real` / `real` and `1_fake` / `fake` path components;
  - `is_image`: skips `__MACOSX` and dot files;
  - `inspect_zip`: size, `testzip` CRC check of every member, images per class. It stops on a missing or unreadable archive, or an image count more than 10% from about 26,000;
  - `unpack`: the whole archive, to the local disk;
  - `unreadable`: PIL `verify` and `load` of every image;
  - `items`.
- **`src/eviground/eval/subsets.py`:**
  - `p1_test_subset`: per generator and label, outside the dev split; stops when a stratum is short;
  - `p4_subset` with `balanced_counts`: half each, else proportional, with largest remainders;
  - both use the seeded SHA-256 rank of `devsplit` and `file_list_sha256`.
- **Notebook 12 `12_eval_subsets.ipynb`** (generator), CPU, estimated 0.1 CU:
  1. pre-flight (zip shortcut, dev split, CP1 manifest);
  2. copies the zip to the local disk (one read through Drive), checks it, unpacks it in full locally and opens every image. It stops and asks on any failure, unreadable image or unlabelled image;
  3. freezes `runs/checkpoints/eval_subsets.json`. An existing file is reused only when both hashes match; otherwise it stops;
  4. writes `eval_subsets_record.json`: counts per class available and frozen, the zip check, timings, compute;
  5. writes the compute record and releases the runtime.
- **CI:** a step for the notebook 12 dry run.

### Checks (sandbox, Python 3.13.18, CPU)
- **5 new tests in `tests/test_eval_subsets.py`:**
  - labels from paths;
  - zip check, unpack and read on a fake archive, including one corrupted image found;
  - stops on a missing archive, a non-zip and a wrong count;
  - P4 balanced (10/10), proportional (49 → 20 + 29) and deterministic under reordering;
  - P1 excludes the dev split, takes `per_label`, has a stable hash, and stops on a short stratum.
- `ruff check src tests`: clean. `python -m pytest -p no:warnings`: 450 passed. `python -m eviground.utils.repo_audit --history`: ok.
- **`EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/12_eval_subsets.ipynb`:** passed.
  - Fake zip of 40 images (24 real, 16 fake) → P4 10 + 10.
  - Fake manifest (2 generators × 5 per label, 1 per stratum in the dev split) → P1 12 images.
  - Then release (dry).

### Review fixes (PR #46)
- **Codex P2:** on a rerun, the frozen file was reused when its stored hash equalled the new one, without recomputing that hash from the stored uid list.
  - New `subsets.intact()` recomputes the hash and image count.
  - Cell 4 reuses the file only when it is intact and its hash matches; otherwise it stops and asks.
  - A new test covers an altered uid list.
- **SonarCloud S7519 (minor):** `dict.fromkeys` in `balanced_counts`.
- **Codex P2 (second finding):** a whole missing generator/label stratum was not caught, so P1 could have been frozen below 20,000.
  - `p1_test_subset` now requires both labels for every generator in the P1 rows after the dev split is removed.
  - With `generators=8` (the real run), it also checks the generator count.
  - A new test covers a removed stratum and a wrong count.
- **Rerun:** ruff clean; 452 passed; the notebook 12 dry run passed.

### Not run here
- The real `Chameleon.zip` (Drive only) and the real manifest: notebook 12 on a Colab CPU runtime.

## PR M — the CP4 record — 2026-11-18

**Docs only:** `docs/CHECKPOINTS.md` (the CP4 entry and its status row) and `docs/EXPERIMENT_LOG.md` (the G2 ledger row).

- **Every number comes from the result files on Drive:**
  - `cp4_generalization_gate_seed42.json`;
  - `cp4_generalization_gate_seed42_ctx.json`;
  - `localization_diagnosis_seed42{,_ctx}.json`;
  - `cp4_fix_training_health_unified_seed42.json`;
  - `cp4_fix_sampler_check.json` (from notebook 10's printout).
- **The G2 gate run** (notebook 09, `RUN_EXPERIMENT = 'eviground_ctx'`, `P1_ENABLED = False`, `P2_ENABLED = True`,
  commit `ff1b14d`, 0.370 h, 0.72 CU):
  - `p1_source` names the saved CP3 file (`p1_rerun: false`), and G1/RF equal it;
  - G2: edited IoU 0.5101, FAIL;
  - the patch-probe numbers equal the mistaken run's, as expected: the probe is trained once and cached.
- **CP4 compute in all:** 15.84 CU, summed from the seven runs in the CP4 entry's table.

### Checks
- `ruff check src tests`: clean. `python -m pytest -p no:warnings`: 445 passed. `python -m eviground.utils.repo_audit --history`: ok.

## PR N — lean GPU sessions: pre-flight, compute record, runtime release — 2026-11-18

**Why:** compute is the binding constraint, with 45 CU left (Dhanush, 18 Nov). Every GPU notebook now:
1. checks and prints the run before any staging;
2. writes its own compute record;
3. releases the Colab runtime as its last step.

**New `src/eviground/utils/session.py`:**
- **`preflight(what, paths, est_cu, cu_left, expect)`**
  - prints the run's settings, the estimated CU and the balance after it;
  - stops (`PreflightError`) when an input is missing, a setting differs from `expect`, or the estimate exceeds `cu_left`.
- **`finish(folder, t0, cu_per_hour, name, extra)`** writes the session's compute record: wall hours × rate, the commit and the GPU.
- **`release(dry)`** calls `google.colab.runtime.unassign()`. It does nothing outside Colab and in the dry run, and says so.

**Notebooks 06, 08, 09 and 11** (via their generators) get:
- **a cell 2b, pre-flight.** It sets `CU_LEFT = None` (enter the Colab balance to enforce it) and `EST_CU`, both estimated from measured runs:
  - 06: 5.9 unified / 3.0 detection;
  - 08: 1.6;
  - 09: 2.8 with P1, 0.3 P2-only;
  - 11: 5.2.
- **a last cell** that writes the session's compute record, then releases the runtime:
  - `compute_session.json` in the run folder for 06 and 11 (the trainer's `compute.json` still holds GPU training hours);
  - `runs/cp4/compute_*.json` for 08 and 09.
- **`FIGURES_ONLY` instructions** now say to run cells 1, 2 and the last two cells.

### Checks (sandbox, Python 3.13.18, CPU)
- **7 new tests in `tests/test_session.py`:**
  - the printout and the balance after the run;
  - the three refusals;
  - the `compute.json` fields;
  - `release` in the dry run, outside Colab, and with a fake `google.colab` (`unassign` called once).
- `ruff check src tests`: clean. `ruff format --check src tests`: 90 files already formatted.
- `python -m pytest -p no:warnings`: **445 passed**.
- `python -m eviground.utils.repo_audit --history`: ok.
- **`EVIGROUND_DRY_RUN=1 pytest --nbmake`** on notebooks 06, 07, 08, 09, 10 and 11: all passed.
  - In 06, 08, 09 and 11 the release cell asserts that it was reached (`'dry'`).
- **The one-off P2-only dry run of notebook 09** (`P1_ENABLED = False`, a `_ctx` model) was run again with the new cells: passed.

### Review fixes (PR #44)
- **SonarCloud S3358 ×3 (major):** nested conditional expressions in the pre-flight cells of 06 and 09. They are now plain `if` statements.
- **Codex P2:** notebook 09's pre-flight now also requires `runs/checkpoints/cp1/manifest_cp1.csv.gz` and `cp1_data_integrity.json`, which cell 4 reads. A missing CP1 file now stops the run before staging, not after.
  - Checked by running the cell's code outside the dry run, with both files absent: refused, naming both.
- **Rerun:** ruff clean; 445 passed; the dry runs of 06, 08, 09 and 11 passed; the P2-only dry run of 09 passed.
- **SonarCloud S1192 (critical), from the fix above:** notebook 09 now repeated `'manifest_cp1.csv.gz'` 3 times. It is now the constant `CP1_MANIFEST`, set in cell 2. Rerun: the dry runs of 08 and 09 and the P2-only dry run passed; 445 passed.

### Not run here
- `runtime.unassign()` on a real Colab runtime. It is checked only against a fake module.

## CP4 PR L — a P2-only gate run, and a guard after a run on the wrong model — 2026-11-18

**What happened.** Notebook 09 ran on 18 Nov with `P2_ENABLED = True`, but `RUN_EXPERIMENT` was left at `None` (commit `33707a3`, L4, 2.232 h, 4.33 CU).

- **It scored P2 on the CP3 unified model.** Dhanush had chosen the fixed model for G2 on 17 Nov, from validation data only, before any P2 number existed.
- **Evidence:**
  - the unified model's P1 raw mean is 0.8809895833, identical to the CP3 run;
  - the file was written without the `_ctx` suffix, so it replaced the 17 Nov `cp4_generalization_gate_seed42.json`. Its G1 and RF values are identical.
- **These numbers are recorded as a mistaken run, not as the gate.** G2 is still gated once, on the fixed model (Dhanush, 18 Nov).

**CP3 unified model on P2 dev** (486 images; whole images, 0.5 threshold):

| | model | patch probe |
|---|---|---|
| edited IoU (G2 metric) | **0.4904** (bar 0.80) | 0.1392 |
| two-class mIoU | 0.5362 | 0.3761 |
| over-marking, share of pixels | 0.1948 | 0.0331 |
| misses, share of edited pixels | 0.2648 | 0.8489 |
| pixel AUC | 0.7943 | 0.6530 |
| mean boundary F1 | 0.2306 | 0.0624 |
| edited images called edited | 0.9794 | 0.9671 |
| held-out-editor pixels only: edited IoU | 0.4162 | 0.1355 |
| real base (257 images): edited IoU | 0.7518 | 0.1574 |
| real base, held-out-editor pixels: edited IoU | 0.6779 | 0.1287 |

**Changes:**

- **Notebook 09 (via its generator), new setting `P1_ENABLED` (default `True`).** With `False`:
  - P1 is not staged or scored;
  - only the unified model and its probe are loaded;
  - the P1 table of the detection model and its probe comes from the saved CP3 gate file;
  - G1 and RF are recomputed from those rows, so they are identical;
  - the report records `p1_source` (file, its `utc`, its commit, `p1_rerun: false`);
  - the unified P1 row and the false-"edited" rate are left out. The saved ones belong to the CP3 unified model.
- **Guard `cp4.check_run_settings`:** a real run with `P2_ENABLED = True` stops unless `RUN_EXPERIMENT` is exactly `'eviground_ctx'` (`cp4.FIXED_EXPERIMENT`). Review fix, Codex P2: the first version accepted any non-empty value, so `'eviground'` (the CP3 run) would have passed.
- **`cp4.p1_from_report`:** returns the saved P1 table, the probes and the false-edited rates. It raises if a detection or detection-probe version is missing.

### Checks (sandbox, Python 3.13.18, CPU)
- **3 new tests in `tests/test_cp4_eval.py`:**
  - a reused P1 table gives the same G1 and RF as the saved checks;
  - a missing harmonized row raises;
  - the guard refuses P2 on the CP3 model in a real run, and allows it with the fixed model or in a dry run.
- **One-off dry run of notebook 09** (fake world, tiny models, script
  `/tmp/claude-0/-home-user-Paper1/d56e2aef-4c2a-5b25-a13a-a24272263c08/scratchpad/p1off/run.py`, not committed):
  1. a full pass with P2 on;
  2. then a P2-only pass with `P1_ENABLED = False` on a `_ctx` copy of the model, writing `cp4_generalization_gate_seed42_ctx.json`.
  - Asserted: `p1_source` points to the first file; G1 and RF are equal; only the detection rows are kept; P2 is scored; there are no P1 statistics. Passed.
- `ruff check src tests`: clean. `ruff format --check src tests`: 88 files already formatted.
- `python -m pytest -p no:warnings`: 438 passed (the guard test now also rejects `'eviground'`).
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 08 and 09: 2 passed. Notebook 08 is unchanged.

## CP4 PR K — notebook 09 restored after a Colab save; the CP4 runs recorded — 2026-11-17

**Why:** on main, `ff36b71` "Created using Colab" (17:55 UTC) saved notebook 09 with its outputs.
That copy was opened before PR #41, so it removed PR #41's `RUN_EXPERIMENT` / `RUN_TAG` cells. Notebook 09
is now generated again from its generator script, with no outputs. Its cell sources are identical to
`d41bba6`, the version PR #41 merged. Notebook 11 was not touched by the save. A second Colab save, `722b31d` (22:02 UTC), did the
same to notebook 08: it added outputs and the cell-2 edit `RUN_EXPERIMENT = 'eviground_ctx'`. Notebook 08
is generated again too, so both notebooks equal `d41bba6` (`RUN_EXPERIMENT = None` by default). Please do not use
"Save a copy in GitHub" in Colab: the results go to Drive.

### Real runs (Colab; numbers from the result files on Drive; validation data and the P1 development split only)
- **Notebook 10, sampler shortcut check** (CPU, commit `616bec5`, 294 s): epochs 0–2, 72,000 draws per sampler.
  - **Metadata classifier, real vs not real:** 0.5122 (CP3) → 0.5120 (fix); 3-class 0.3443 → 0.3434. CP1 run 5 for reference: 0.5071 / 0.3469.
  - **Quality-75 share (`q75_any`):** real 0.3727 → 0.3727; generated 0.3700 → 0.3716; edited 0.3679 → 0.3659.
  - **Stored image size, real vs not real** (a diagnostic, not a criterion): 0.7832 → 0.7833.
  - **Classes:** edited 24,000 → 22,802; generated 24,000 → 25,198.
  - **Context draws:** 2,608 stay edited; 1,198 are edit-free and relabelled generated.
  - **Generated-class sources:** GenImage 0.6666 → 0.6349, SID-Set 0.3334 → 0.3175, DiffSeg30k 0 → 0.0475.
  - **Verdict:** `material changes: none`.
- **Notebook 09, CP4 gate on the CP3 models, P2 off** (L4, commit `616bec5`, 1.405 h, 2.73 CU). **G1 FAIL; RF PASS.**
  - **Mean accuracy over 8 generators** (detection model / detection probe / unified model / unified probe):
    - raw: 0.7846 / 0.8136 / 0.8810 / 0.8953;
    - harmonized: 0.7771 / 0.8015 / 0.8642 / 0.8777.
  - **Mean AP, raw:** 0.9479 / 0.9194 / 0.9692 / 0.9563.
  - **Detection model, raw:** real accuracy 0.994–0.998 on every generator. Generated accuracy:
    - ADM 0.122, GLIDE 0.248, VQDM 0.354, BigGAN 0.423, Midjourney 0.473;
    - Wukong 0.968, SD v1.4 0.999, SD v1.5 0.998.
  - **RF:** only SD v1.5 (0.9969) and Wukong (0.9813) reach 0.98.
  - **Unified model, real P1 images called edited:** 3.85% overall.
    - By tile count: 1: 0.68%; 2–4: 3.05%; 5–9: 9.07%; 10–16: 15.56%; 17+: 21.64%.
  - **Probes:** class token L2 0.001 (validation AUC: detection 0.9965, unified 0.9784); patch probe validation mask F1 0.2471.
  - **Dhanush's decision:** the one fix stays with the unified model. G1 is recorded as failed, and the detection claim is narrowed.
- **Notebook 11, retrain of the unified model with context crops** (L4, commit `ff36b71`; training 2.0126 GPU h, 3.421 CU at 1.7 CU/h; session 3.056 h, 5.2 CU).
  - **Run:** best epoch 15 of 20, stopped early.
  - **Health checks:** S1, S2a (macro AUC 0.997864), S2b, A1 (step 25 / 50), A3 and A4 (mask F1 0.758295) passed.
  - **A2 FAILED:** real-crop marked area 0.010302 against a bar of ≤ 0.01; any mark 0.10532.
  - **Against CP3's best (validation crops):**
    - pooled edited IoU 0.67661 → 0.668879;
    - generated crops with any mark: GenImage 0.0153 → 0.0273, SID-Set 0.0283 → 0.0638;
    - real crops with any mark: 0.0815 → 0.1053.
  - **Dhanush's decision:** run notebook 08 anyway; the A2 bar is unchanged.
- **Notebook 08 on the fixed model** (L4, commit `ff36b71`, 0.652 h, 1.26 CU), against CP3's diagnosis (c8ed936, 0.827 h, 1.6 CU):

| whole image, 1,177 edited images | CP3 | fix |
|---|---|---|
| edited IoU (DiffSeg30k mIoU) | 0.3946 | 0.4335 |
| over-marking, share of all pixels | 0.2886 | 0.2148 |
| misses, share of edited pixels | 0.1908 | 0.2276 |
| AI base: IoU / over-marking / misses | 0.3303 / 0.3589 / 0.1945 | 0.3689 / 0.2628 / 0.2425 |
| real base: IoU / over-marking / misses | 0.7209 / 0.0486 / 0.1821 | 0.7082 / 0.0507 / 0.1923 |
| real images called edited (5,545) | 0.0893 | 0.1053 |

  - **Real images called edited, by source** (fix): SID-Set 0.283, DiffSeg30k 0.190, COCO 0.079, GenImage 0.049.
  - **Interior tiles** (model; 1,065 tiles that are ≥ 90% edited, holding 30.3% of the edited pixels):
    - 10.7% are called fully generated; mean P(generated) is 0.147;
    - 33.4% of their edited pixels are missed, against 18.1% elsewhere;
    - in the 40–100% bin: 35.6% against 17.1%.
  - **Threshold sweep:** best 0.4458 at 0.70. The gate stays at 0.5.
  - **Patch probe:** edited IoU 0.1527.
  - **Dhanush's decision:** G2 is gated once, on the fixed model.

### Checks (sandbox, Python 3.13.18, CPU)
- `ruff check src tests`: clean. `ruff format --check src tests`: 88 files already formatted.
- `python -m pytest -p no:warnings`: 435 passed.
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 08 and 09: 2 passed.
- `git diff d41bba6 -- notebooks/`: empty, so notebooks 08 and 09 are back to the version PR #41 merged.

## CP4 PR J — the retrain notebook and the evaluation of the fixed model — 2026-11-17

Prepared while PR I waits. **The L4 retrain runs only after Dhanush's go-ahead,** and after notebook 10 found no material change.

- **New notebook 11 `11_cp4_fix_unified.ipynb` (L4),** built from notebook 06's generator:
  - trains unified seed 42 with `data.context_crop_share = 0.3333` into `runs/eviground_ctx_unified_seed42/`; CP3's model is not touched, and CP3's A1 record is reused;
  - runs the CP3 checks on the run;
  - writes `cp4_fix_training_health_unified_seed42.json`;
  - **compares with CP3's `best.pt` on validation crops** (`fixcompare.compare_runs`, each run at its best epoch): macro AUC, pooled edited IoU, mask F1, SID-Set and GenImage generated crops called edited or with any mark, and real crops with any mark.
- **Notebooks 08 and 09:** `RUN_EXPERIMENT = 'eviground_ctx'` selects the fixed unified model (detection stays CP3's), and every result file then gets the suffix `_ctx`, so the CP3 results are kept.
  - Notebook 08 with the fixed model also compares its whole-image numbers with the CP3 diagnosis (`fixcompare.compare_diagnoses`): edited IoU, over-marking and misses, overall, on AI bases and on real bases.
- **Interior tiles** (Dhanush's prompt; measured on the fixed model only):
  - the tiled evaluator keeps each tile's class probabilities and box;
  - `locdiag.interior_record` takes the tiles at least 90% edited, counts how often the tile's class head calls them fully generated and their mean P(generated), and splits the missed edited pixels into those inside such tiles and the rest;
  - `interior_summary` pools them, overall and by edit-size bin, for the model and the probe.

### Results (sandbox: Python 3.13.18, CPU)
- `ruff check src tests`: clean. `ruff format --check src tests`: 88 files already formatted.
- `python -m pytest -p no:warnings`: **435 passed**, 4 of them new, in `tests/test_fix_eval.py`:
  - `interior_record` on a hand-made map (1 interior tile called generated; 16 of 32 interior pixels missed; 0 of 16 others);
  - pooling;
  - `compare_runs` at each run's best epoch;
  - `compare_diagnoses` by base.
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 06, 07, 08, 09, 10 and 11: all passed. The 08 dry run asserts the interior block for both heads.

### Review fix (PR #41)
- SonarCloud S1192 (critical; the quality gate passed): notebook 11 repeated the literal `'metrics.csv'` 3 times. It is now a constant, `METRICS`, set through the generator.
- Rerun: ruff clean; 435 passed; audit ok; the notebook 11 dry run passed.

### Not run here
- Notebook 11 (L4, ≈ 4.5–5.5 CU estimated): after Dhanush's go-ahead.
- Then notebook 08 and notebook 09 with `RUN_EXPERIMENT = 'eviground_ctx'`.

## CP4 PR I — the one CP4 fix: context crops on AI-generated bases, and its shortcut check — 2026-11-17

**Why:** the localization diagnosis (notebook 08 on validation data of the training editors, CP3 unified seed 42) found:
- whole-image edited IoU 0.395 (crops: 0.677);
- **over-marking** on 28.9% of all pixels; misses on 19.1% of edited pixels;
- by base: AI-generated bases IoU **0.330** with 35.9% of pixels over-marked, real-photo bases **0.721** with 4.9%.

Every edited training crop held part of the edit (`crops.edited_crop`), so the model never saw an AI-generated base far from an edit.

**Dhanush's decision (17 Nov 2026):**
- For edited images with an AI-generated base, **1/3 of the training draws take a crop anywhere** in the image.
- A draw is kept only if its edited share is **exactly 0** (labelled `generated`, with an all-zero mask) or **1–90%** (`edited`, with the true mask). A share in (0, 1%) or above 90% is redrawn, up to 10 tries, and the CP3 rule is the fallback.
- Every other edited crop keeps the 1–90% rule. Real-base edits are unchanged.
- The interior-tile hypothesis (crops inside an edit) is not part of the fix: notebook 08 found misses of 19.1%, and 21.1% in the 40–100% bin.
- The gate stays at threshold 0.5.

Code:
- **`crops.context_crop`:** a uniform window, accepted at share 0 or within (1%, 90%).
- **`datasets.ForensicDataset(context_share=…)`:**
  - **Only AI-base edited rows** take context crops; the generator draws nothing extra when the share is 0, so CP3's plans are identical.
  - A crop spec carries the tag `context` or `context_generated`. `__getitem__` relabels `context_generated` as `generated`, with `has_mask` False and a zero mask.
  - **The loader re-checks the edit mask** and raises `ContextCropError` if a crop labelled generated holds any edited pixel.
  - `context_counts()` counts both kinds.
- **`configs/base.yaml`:** `data.context_crop_share: 0.0`, so CP3 is unchanged. The trainer passes the share through and logs the counts per epoch.
- **New `data/sampler_check.py`** reruns the CP1 shortcut checks on the trainer's own draws:
  - the metadata lookup table on CP1's criterion features (format, JPEG quality, crop size), real vs not real (≤ 0.60) and 3-class;
  - the stored image size, as a diagnostic;
  - the quality-75 share per class and per class/source;
  - the source mix per class.
  - `compare` flags a material change: real vs not real above 0.60; a balanced-accuracy change above 0.02; a quality-75 or class/source share change above 0.03.
- **New notebook 10 `10_cp4_fix_sampler_check.ipynb`** (CPU, before any GPU time):
  - builds epochs 0–2 of the CP3 sampler and of the fixed one (72,000 draws each);
  - runs and compares the checks, showing CP1 run 5's numbers for reference;
  - writes `cp4_fix_sampler_check.json`;
  - **stops** on a material change.

### Results (sandbox: Python 3.13.18, CPU)
- `ruff check src tests`: clean. `ruff format --check src tests`: 86 files already formatted.
- `python -m pytest -p no:warnings`: **429 passed**, 7 of them new, in `tests/test_data_pipeline.py`:
  - `context_crop` keeps edit-free windows and 1–90% windows, and refuses fully edited ones;
  - share 0 gives the CP3 plan, unchanged;
  - context crops only on AI-base edits, edit-free ones labelled generated with a zero mask, seeded;
  - the share is near the one requested;
  - a crop wrongly tagged edit-free is refused (`ContextCropError`);
  - per-class counts: edited loses exactly the edit-free draws and generated gains them;
  - the sampler check and its comparison (DiffSeg30k appears in the generated class only with the fix).
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 06, 08, 09 and 10: all passed. In the fake world, notebook 10 found no change between the samplers apart from the context crops. Its stop fires on the fake world's own metadata score (0.84), as intended for real data above 0.60.

### Review fixes (second push)
- **SonarCloud, 2 critical (S3776):** `sampler_check.compare` was split into `_q75_changes` and `_material`, `_diffs` into `_number` and `_mapping` (third push), and the trainer reads the share through `_context_share`. The other 6 findings were minor: one `endswith` with a tuple, and 5 composite test assertions split.
- **Codex P2, resume:** a run resumed with a different `data.context_crop_share` would rebuild a different epoch plan and skip the wrong draws. `Trainer._check_run_settings` now compares profile, seed and the share with the checkpoint's config (a CP3 checkpoint without the setting counts as 0) and stops with `CheckpointMismatch`. New test.
- **Codex P2, quality 75 by source:** the comparison also covers `per_class_source`, for pools present on both sides. New test: the sources move while the class total stays the same, and the change is flagged.
- **Materiality of the source mix:** the fix moves DiffSeg30k into the generated class by design, which also dilutes the other generated sources.
  - Materiality is now judged on the mix without the edit-free context crops (`source_mix_without_context_generated`).
  - The designed change is reported apart (`source_share_change_designed`).
- After the fixes: **431 passed**; ruff clean; the audit ok; the notebook 06 and 10 dry runs passed.

### Not run here
- Notebook 10 on the real shards: a CPU runtime, a shard copy, then 2 × 72,000 draws.
- The retrain (notebook 11, PR J): after Dhanush's go-ahead.

## CP4 — the BR-Gen subset is frozen (notebook 07 on Colab, CPU) — 2026-11-17

Run on main at `c8ed936`, 09:59–10:00 UTC, 0.003 CU. Results are read from Drive: `cp4_prep.json`, `brgen_subset.json` and `brgen_sample_check.json` in `runs/checkpoints/`.

- **Development split:** reused and verified, `1e0792a8…` (P1 20,000 of 100,000; P2 486 of 2,434).
- **Archive:** 13 volumes kept, 0 downloaded. The listing found 150,000 image–mask pairs, 5,000 in each of the 30 method × region × source groups, with 0 unpaired and 0 ambiguous.
- **Subset `bebb86c5c9d20ba8ccb603307011e9e07eb0f52a2fe85b894a90c3ce2eebf0bb`:**
  - 6,000 forged images: 1,200 source images × 5 methods, 200 per region × source group;
  - every source image has all 5 methods, and its 1,200 masks are shared by the methods;
  - distinct source ids: COCO 391, ImageNet 396, Places 389.
- **Training overlap, by id:** ImageNet **168** of 5,000 BR-Gen source images are among the GenImage training reals. All 168 are excluded, and each ImageNet group still got 200. COCO and Places: 0.
  - BR-Gen's own lists name its images the way our data does (`000000120836.jpg`, `n01797886_20962.JPEG`, `h-hangar-outdoor-00002800.jpg`). The ImageNet list also gives 168 matches.
  - Training rows carrying a COCO or ImageNet id: COCO 4,837 of 4,837; GenImage 39,912 of 79,842 (the real half); SID-Set 50 of 19,868; DiffSeg30k 0 of 16,682. DiffSeg30k (COCO-based) therefore cannot be checked by id, so P3 is reported per source and on ImageNet + Places alone.
- **Sample check: pass.**
  - 20 images extracted in 33.3 s, 375–949 px wide, 4–12 tiles each.
  - Every mask has the image's size and the values {0, 255}. Coverage is 1.0, and stitching returns the mask exactly.
  - The edited share ranges from 4.9% to 99.85%, with a median of 46.6%. Background edits can cover almost the whole image, so P3 is also reported per region.

On 10:06 UTC a Colab save committed this notebook with its outputs to main (`e281e79`); its cell sources were unchanged. PR H's merge keeps the generated notebook without outputs, as for every notebook here. The numbers above are the record.

The same happened to notebook 08 at 10:59 UTC (`cd6d6b0`: 25 outputs and Colab metadata, cell sources unchanged), and it was resolved the same way. Its results are in `localization_diagnosis_seed42.json` on Drive. The one warning in its outputs (a read-only NumPy array in `tiled.py`) is fixed: the tile is copied.

## CP4 PR H — seaborn figures, drawn from the saved result files — 2026-11-17

Dhanush asked for seaborn in the notebooks' plots, for a clearer view.

New `src/eviground/viz.py`:
- **Drawn only from saved files.** Figures come from what a notebook already wrote to Drive (its result JSON and per-image CSV), never from objects in memory. A figure can therefore be redrawn or restyled on a CPU runtime without running a model again.
- **Theme:** seaborn `whitegrid` with the colour-blind palette. Legends sit below the axes, and every figure is saved as PNG (150 dpi) and PDF.
- **Tidy rows:** `curve_rows`, `sweep_rows`, `per_image_rows` (a multi-turn image gives one row per editor) and `p1_rows` need only the standard library.
- **Notebook 08 figures** (`diagnosis_figures`):
  - the CP3 training curves (mask metrics, with the G2 bar; marks on real crops, symlog);
  - the validation threshold sweep (pixel metrics; image calls; dotted line at 0.5);
  - per-image IoU by edit size, beside the 16 × 16 and 64 × 64 resolution ceilings, with n per bin;
  - per-image IoU by editor and by base type, EviGround against the CLIP patch probe.
- **Notebook 09 figure** (`p1_figure`): P1 accuracy per generator, raw and harmonized side by side, all four heads, with the 0.721 bar.

Notebooks (via their generators):
- **Notebook 08:** the two matplotlib blocks in cell 6 are removed. A new cell 8 draws the figures from `localization_diagnosis_seed42.json` and `diagnosis_edited_seed42.csv` into `runs/cp4/figures/`. To redraw: cells 1, 2 and 8 on a CPU runtime.
- **Notebook 09:** a new cell 9 draws the P1 figure from `cp4_generalization_gate_seed42.json`.
- **Notebook 07:** the sample check copies the decoded image (`np.array`), which silences PyTorch's "array is not writable" warning. The frozen BR-Gen subset is unaffected.

Dependencies: `seaborn>=0.13` in `train` (already preinstalled on Colab) and in `dev` (CI draws the figures in the tests and in the notebook dry runs).

Carried over: the split composite assertion in `tests/test_shared_trunk.py` (SonarCloud S9073 on PR #37).

### Results (sandbox: Python 3.13.18, CPU, seaborn 0.13.2, matplotlib 3.11.2, pandas 3.0.5)
- `ruff check src tests`: clean. `ruff format --check src tests`: 85 files already formatted.
- `python -m pytest -p no:warnings`: **422 passed**, 5 of them new, in `tests/test_viz.py`:
  - curve rows skip missing epochs;
  - sweep rows name the heads and keep boundary F1 only where it was measured;
  - per-image rows explode editors and skip missing values;
  - P1 rows;
  - all five figures are drawn from a saved and re-read JSON, as PNG + PDF.
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 07, 08 and 09: all passed. 08 wrote 4 figures and 09 wrote 1, each as PNG + PDF.
- I checked the figures by eye on synthetic data: no legend covers data, epochs are whole numbers, and the real-crop panel has no negative ticks.

### Review fixes (second push)
- **SonarCloud S3776, critical, ×3:** `sweep_rows`, `per_image_rows` and `diagnosis_figures` had cognitive complexity 16–17 (limit 15). They are split into one helper per row kind (`_sweep_point`, `_image_rows`) and one per figure (`_curves_figure`, `_sweep_figure`, `_size_figure`, `_editor_figure`). The output is unchanged: the 5 figure tests pass.
- **Codex P2:** the "redraw on a CPU runtime" instruction failed, because cell 2 required a GPU.
  - Cell 2 of notebooks 08 and 09 now has `FIGURES_ONLY = False`. Set to True, it skips the GPU check, and cell 3 stops at once, so no model runs.
  - Checked without a GPU: cell 2 with `FIGURES_ONLY = True`, then the figures cell, on the saved files of the 08 dry run copied to a stand-in Drive. All 4 figures (PNG + PDF) were drawn, and cell 3 stopped with its message.
- Full checks after both fixes: 422 passed; 08 and 09 dry runs passed; ruff clean; audit ok.

### Not run here
- The figures on real results: they come with notebook 08's run, and with notebook 09's.

## CP4 fix — the CLIP probe training kept every batch's tokens on the GPU — 2026-11-17

**What went wrong:** notebook 08, cell 4, stopped with `CUDA out of memory`: 21.48 GiB was allocated on the 22 GiB L4 before the trained model was even moved to it. No probe was saved (`runs/cp4/probes/` is empty).

**Cause:**
- `SemanticBranch.joint_tokens` returns the class token as `post[:, 0]`, a view of the whole `(B, 257, D)` token tensor.
- Under bf16 autocast, LayerNorm outputs fp32, so `cls.float()` in `clip_probe._features` returned that same view.
- `train_probes` keeps every batch's class token in `xs`. Each one therefore held all the batch's tokens: about 67 MB per batch of 64 with the real CLIP (D = 1,024) instead of 0.26 MB.
- 3 epochs × 24,000 crops is 1,125 batches, about 76 GB in all. The L4 fills after about 320 batches.

**Fix:** `_features` copies the class token out (`cls.float().clone()`). `train_probes` then keeps about 0.3 GB of class tokens in all. `collect_cls` and notebook 09 use the same function.

### Results (sandbox: Python 3.13.18, CPU)
- New test `test_probe_features_do_not_keep_the_whole_token_tensor` in `tests/test_shared_trunk.py`. On the tiny CLIP it failed before the fix (a 1,024-byte class token held 263,168 bytes of storage) and passes after it.
- `ruff check src tests`: clean. `python -m pytest -p no:warnings`: **417 passed**. `repo_audit --history`: ok. `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 08 and 09: passed.

### Not run here
- The L4 run: restart the Colab session (to free the 21.5 GB), then run notebook 08 from the top.

## CP4 PR G — the paired BR-Gen subset, without the training overlap — 2026-11-17

Notebook 07, rerun after PR F, listed BR-Gen correctly: **150,000 pairs**, 5,000 in each of the 30 method × region × source groups, 0 images without a mask, 0 masks without an image, 0 ambiguous (18.8 s, 0.001 CU, commit `0025262`). The dev split was reused unchanged (`1e0792a8…`).

Dhanush's decisions (17 Nov 2026):
- **Paired subset, 200 per group:** in each of the 6 region × source groups, 200 source images, each with all 5 methods' forged files. That is 1,200 source images and 6,000 forged images; the methods are compared on the same images.
- **P3 reports pixel metrics only:** BR-Gen ships no real images, so there is no image-level accuracy or AUC.
- **Training overlap checked and left out:** a source image whose id appears in the CP1 `train_pool` is not chosen.
- **Kept COCO, reported by source:** DiffSeg30k's real bases are COCO photos (its paper, Sec. 3.1), but its file names are sequential numbers (e.g. `diffseg30k/train_copy/00034156.image.png`), so a COCO overlap with them cannot be checked by id. P3 is therefore reported per source, and also on ImageNet + Places alone.

Code (`data/brgen.py`):
- `source_id` reads the source image's id from a forged or mask path (`000000232161_background.png` → `000000232161`).
- `training_ids` collects ids from the manifest rows with role `train_pool`: each uid's file stem, plus every COCO id (12 digits) or ImageNet training id (`n########_#`) in the uid. `coverage` counts, per source, the rows whose uid carries such an id, so a source that cannot be checked by id shows 0.
- `overlap` counts BR-Gen source images per source whose id is in training.
- `paired_subset` ranks the source images of each region/source group by a hash of (seed, salt, group, id). It skips images that miss a method or are in `exclude`, takes the first `per_group`, and returns all methods' files of those images, their masks and the SHA-256 of the sorted list.
- `released_paths(..., ids=...)` lets tests and the dry run choose source-image ids.

Notebook 07:
- Cell 2: `FREEZE_BRGEN = True`, `SAMPLE_CHECK = True`, `PER_GROUP = 200`, plus `P3_METRICS` and `P3_REPORTING`, which are stored in the subset file.
- Cell 7 freezes the paired subset once, into `brgen_subset.json`. It extracts the three source-image lists (text only) and records their entry count, 3 examples and how many match training ids. It records the id coverage per training source and the overlap per BR-Gen source. It stops if a group falls short of 200.
- The dry-run archive holds one COCO and one ImageNet id that the fake training pool also holds. The dry run asserts that both are found and left out.

### Results (sandbox: Python 3.13.18, CPU)
- `ruff check src tests`: clean; `ruff format --check src tests`: 83 files already formatted.
- `python -m pytest -p no:warnings`: **416 passed**, 3 of them new, in `tests/test_cp4_prep.py`:
  - every chosen source image has all 5 methods, the subset is seeded and independent of input order, and the masks follow;
  - an image missing a method is skipped;
  - overlap by COCO and ImageNet id is found and excluded, a `test_P7` COCO image does not count as training, and a source whose names carry no id shows 0 coverage.
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 07, 08 and 09: all passed. In 07, the overlap was found as 1 COCO and 1 ImageNet image, and each COCO and ImageNet group excluded 1; the sample check passed.

### Not run here
- The real freeze: notebook 07 must be rerun on a CPU runtime. It extracts 3 text lists and about 20 sample images, a few minutes and under 0.05 CU.
- How many training rows of each real source carry ids, and how many BR-Gen source images overlap: known only after that run.

## CP4 PR F — BR-Gen's released layout is parsed and paired — 2026-11-17

Notebook 07 listed the complete archive: 13 volumes, 180,003 files, 68,555,453,840 bytes, read in 21.5 s for 0.015 CU. It paired **0** images with masks, because the code expected the README layout (`Au/`, `Tp/`, `Gt/`, `<stem>_mask.png`). The released archive, read from `brgen_listing.json` and `brgen_members.txt.gz` (file names only), is laid out like this:

- **Forged images:** `BR-Gen/Forged/<Method>/<Region>/<Source>/<stem>_<region>.png`. There are 5 methods (BrushNet, LaMa, MAT, PowerPaint, SDXL), 2 regions (Background, Stuff) and 3 sources (COCO, ImageNet, Places). Each of the 30 combinations holds 5,000 images, so 150,000 in all.
- **Masks:** `BR-Gen/Mask/<Region>/<Source>/<stem>_<region>.png`, 30,000 in all. Each mask is shared by the five methods' images of that file name. Every forged image has its mask.
- **Real images:** `BR-Gen/RealImage/<Source>/<Source>_image_list.txt` holds lists only. The archive contains no real images.

Code (`data/brgen.py`):
- `parse_member` reads the method, region, source and name from a path in the released layout.
- `kind`, `method_of`, `stratum_key` (method/region/source) and `_pair_key` (region/source/name) use those fields. The README layout is still recognised as a fallback.
- `pair_masks` now pairs every image that shares a mask. A key counts as ambiguous only when it has more than one mask.
- `summarize` adds `layout`, `strata`, `mask_strata`, `source_images` and `real_lists`.
- `released_paths` and `fake_listing(layout="released")` build the released layout for tests and dry runs.

Notebook 07:
- The dry-run archive uses the released layout.
- On a released-layout archive, the listing asserts that the number of pairs equals the number of forged images.
- The subset cell no longer needs `STRATUM_DEPTH` for this layout.
- The sample check records a mask/image size mismatch instead of stopping.

### Results (sandbox: Python 3.13.18, CPU)
- `ruff check src tests`: clean.
- `python -m pytest -p no:warnings`: **413 passed**, one of them new, in `tests/test_cp4_prep.py`. It checks parsing, pairing (60 of 60), the 30 strata, per-source counts, the real-image lists and a 1-per-stratum subset. All earlier README-layout tests still pass.
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/07_cp4_prep.ipynb`: passed, with the pairs = forged-images assertion.

### Not run here
- The real listing: notebook 07 must be rerun. It is expected to find 150,000 pairs.

## CP4 PR E — notebook 07 completes the BR-Gen split archive before listing it — 2026-11-17

**What went wrong:** notebook 07 stopped at the BR-Gen listing because the last volume, `BR-Gen_split.zip`, was missing.
- Drive's `04_BR-Gen` held only `BR-Gen_split.z01`–`z08`.
- The Hugging Face repo `clpbc/BR-Gen` (public) publishes `z01`–`z12` at 5,368,709,120 bytes each, plus `BR-Gen_split.zip` at 4,081,350,779 bytes. That `.zip` volume holds the archive's central directory.
- So about 25.5 GB was missing.
- The development split had already been frozen in cell 4: `cp4_dev_split.json` was on Drive at 06:31 UTC.

**Decision (Dhanush, 17 Nov):** fetch only the missing volumes.

Code:
- `download.make_context`: the download context from `ensure_data`, moved into its own function (Drive-quota check, upload-cache wait, network sources created only when needed). `ensure_data` behaves as before.
- `brgen.complete_split(ctx, folder)`:
  - **Which volumes:** takes the published list from the repo and keeps the `BR-Gen_split.*` files.
  - **Kept volumes:** a volume already on Drive at exactly the published size is kept. Its size is checked, but it is not re-hashed (too slow through the Drive mount); the report marks it `size_only`.
  - **Downloaded volumes:** a missing volume, or one of the wrong size, is downloaded after the Drive-quota and upload-cache checks. Its size and SHA-256 are then checked against Hugging Face, and a stamp file is written.
    - A hash mismatch removes the file and raises an error.
    - A wrong-size local copy is replaced, and the report lists it under `replaced`.
  - **Extra files:** files with the same prefix that the repo does not publish are reported and never removed.
- **Notebook 07:**
  - A new cell 5 completes the archive before the listing.
  - The listing cell (now 6) asserts that every published volume is present.
  - In the dry run, the fake BR-Gen zip is served through `FakeSources`, so CI exercises the download path.

### Results (sandbox: Python 3.13.18, CPU)
- `ruff check src tests`: clean.
- `python -m pytest -p no:warnings`: **412 passed** (407 before, plus 5 new in `tests/test_cp4_prep.py`).
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake`: notebook 07 passed; notebooks 00c, 08 and 09 also passed.

### What the new tests cover
- **Missing volumes:** only those are downloaded. A truncated copy is replaced, an unpublished file is reported and kept, and a second run downloads nothing.
- **Hash mismatch:** the file is removed and an error is raised.
- **Too little Drive space:** refused before any download or write, including when there is room for one volume but not for all the missing ones (the total is checked first; Codex P2 on PR #35).
- **No published volumes:** an error is raised.

### Not run here
- The real download, about 25.5 GB from Hugging Face to Drive on a CPU runtime.
- The real 7z listing.

## CP4 PR D — metric definitions checked against the DiffSeg30k paper — 2026-11-17

Checked against DiffSeg30k, arXiv 2511.19111v2, sections 4.1–4.2 (read 17 Nov). The paper says: "The background class is excluded from mIoU and bF1 calculations." So its binary mIoU, including SegFormer's 0.951 on the unseen Glide + Kolors pair (Table 4), is the IoU of the edited class alone.

Dhanush decided on 17 Nov:
- **G2** uses the paper's definition: the IoU of the edited class, pooled over all P2 dev pixels, must be at least 0.80.
- **The paper's image rule** is reported beside ours: an image counts as edited when more than 1% of its pixels are predicted edited.

Code:
- **`eval/tiled.py`**
  - `pooled` reports `miou_diffseg30k` (edited-class IoU) and `miou_two_class` (the mean of the edited and background IoU, our earlier `miou_binary`).
  - `per_image` reports `pred_share`.
  - New `area_rule` and `AREA_RULE = 0.01`.
- **`eval/thresholds.py`:** the sweep reports both mIoU values. `best` defaults to `miou_diffseg30k`.
- **`eval/cp4.py`**
  - G2 reads `miou_diffseg30k` and also records `miou_two_class`.
  - The published GenImage numbers are checked against arXiv 2511.01293, Table 7, SD v1.4 training: ResNet-50 72.1, NPR 88.6, FatFormer 88.9, Ojha 73.3. All four match.
- **`eval/locdiag.py` and `runner.p2_summary`:** both report `called_edited_area_rule` / `image_acc_edited_area_rule`. These come from records that already exist, so there is no extra model pass.
- **`eval/metrics.py`:** a docstring correction only. CP3's `mask_miou_binary` is the two-class mean; the paper's mIoU on crops is `mask_iou_pooled`. No CP3 number changes.
- **Notebook 08**
  - `crop_vs_full` puts both definitions side by side.
  - The sweep plot uses the paper's mIoU.
  - The result file names its definitions and their source.

Not stated in the paper, so these stay ours:
- whether the mIoU is pooled over pixels or averaged per image (we pool, which is mmsegmentation's default);
- the boundary tolerance of bF1 (we use 2 px);
- the boundary rule (we use 4 neighbours, and the image edge is not a boundary).

The paper's bF1 also excludes the background; ours measures only the edited class's boundary, so the two agree on that.

### Results (sandbox: Python 3.13.18, torch 2.15.0, CPU)
- `ruff check src tests`: clean.
- `python -m pytest -p no:warnings`: **407 passed** (406 before, plus 1 area-rule test; the existing gate test gains a two-class-versus-edited-IoU case).
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/08_localization_diagnosis.ipynb notebooks/09_cp4_gate.ipynb`: 2 passed.

## CP4 PR C — validation-only threshold sweep in the localization diagnosis — 2026-11-16

Requested by Dhanush before notebook 08 runs.
- **Diagnostic only:** every headline number and the CP4 gate stay at 0.5. Validation images only (training editors); no test image is read.
- Whether the gate ever uses a validation-chosen threshold is his decision after the diagnosis.

Code:
- **`eval/thresholds.py`:**
  - `confusion_at` and `pixel_sweep` give pooled edited and background IoU, binary mIoU, pooled F1, mean per-image F1, and the FP and FN shares at 0.05–0.95, step 0.05.
  - They are computed from the 1,000-bin score histograms `mask_record` already keeps, so the model is not run again.
  - `boundary_grid` gives the boundary counts at 0.1–0.9, step 0.1, computed on the CPU while the image is evaluated.
  - `image_sweep` sweeps the P(edited) threshold τ: the share of edited images called edited, and the false-"edited" rate on real images, overall and by tile count.
  - `best` gives the arg-max threshold, with the value at 0.5 beside it. `diagnosis_sweeps` builds the result block.
- **`eval/tiled.py`:** `boundary_counts` and `boundary_f1` are split out of `mask_record` and `per_image`. Their results are unchanged.
- **`eval/runner.py`:** `diagnosis_pass` records `p_edited` per head, and the boundary grid when `sweep=True`. Notebook 09's P2 pass sets `sweep=False`, so no threshold is ever measured on test data.
- **Notebook 08:** cell 6 adds the sweep and a plot (`diagnosis_threshold_sweep_seed42.png`). Cell 7 writes `threshold_sweep` into `localization_diagnosis_seed42.json`, and the dry run asserts it.

### Results (sandbox: Python 3.13.18, torch 2.15.0, CPU)
- `ruff check src tests`: clean.
- `python -m pytest -p no:warnings`: **406 passed** (394 before, plus 12 in `tests/test_thresholds.py`).
- `python -m eviground.utils.repo_audit --history`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake notebooks/08_localization_diagnosis.ipynb notebooks/09_cp4_gate.ipynb`: 2 passed.

### What the new tests cover
- The histogram sweep equals direct thresholding of the probability maps at 0.1, 0.25, 0.5, 0.75 and 0.9. At 0.5 it equals the recorded counts.
- `confusion_at` on a hand-made histogram.
- `boundary_counts` at 0.5 equals the boundary counts inside `mask_record`. Boundary F1 appears only at the thresholds on the boundary grid.
- The image sweep: rates and tile-count bins worked out by hand.
- `best`: the arg-max, the value at 0.5, and an empty sweep.
- The shape of the `diagnosis_sweeps` block.

### Not checked
- The DiffSeg30k paper's definitions of binary mIoU and boundary F1 (arxiv.org is blocked from the sandbox).

## CP4 PR B — evaluator, CLIP probes, localization diagnosis, P1 gate — 2026-11-16

From the CP4 plan approved on 16 Nov, with Dhanush's changes:
- P2 is not scored until he has seen the localization diagnosis;
- G1 must also beat the probe on harmonized files;
- the published results are shown for context;
- the false "edited" rate is reported on large real images;
- there is no second fix;
- the shared-trunk features are tested against a plain CLIP.

Code:
- **Shared trunk:** `models/semantic.py` `joint_tokens` / `probe_features`, and `EviGround.forward(probe=True)`.
  - CLIP layers before the first LoRA layer (`first_trainable`, read from the config) run once.
  - The later layers run twice: with LoRA for the model, and with the adapters off for the probe.
  - `adapters_off` sets peft's disable flag only and restores it afterwards; it leaves `requires_grad` untouched.
- `baselines/clip_probe.py`:
  - the class-token logistic regression (L-BFGS, L2 chosen by validation AUC);
  - online patch probes, 6 L2 strengths side by side, chosen by validation mask F1; a patch is positive at ≥ 50% edited, and real and generated crops are all-zero;
  - `ProbeHead`, which gives the model's output contract. The features are computed in bf16 on a GPU; the fits always run in fp32.
- `eval/tiled.py`:
  - tiles of many images are batched by shape; image scores follow the unchanged rule, and masks are stitched;
  - `mask_record` keeps the pixel counts of each image: confusion matrix, AUC histograms, boundary matches at 2 px, and held-out-editor pixels;
  - from those counts: pooled metrics, per-image metrics, histogram AUC.
- `eval/locdiag.py`:
  - grid ceilings (16 × 16 and 64 × 64 per tile, majority per cell, stitched);
  - summaries by edit size, editor and base type;
  - real-image rates by tile count and source.
- `eval/testsets.py`:
  - whole validation images from the shards;
  - P1 and P2 from files, raw or harmonized (GenImage rule, `min_side=None`);
  - P2 masks with the AI base read as 0, plus per-pixel editor values;
  - parallel staging of files.
- `eval/runner.py`:
  - `load_trained` seeds as the trainer does, then checks the frozen-CLIP hash;
  - probe data from the models' epoch plans;
  - the joint predictor;
  - the diagnosis, P1 and P2 passes.
- `eval/cp4.py`:
  - G1, the red flag RF, and G2 (`NOT_RUN` until P2 is enabled; the verdict is then `PENDING_P2`);
  - the P1 table with accuracy and AP;
  - the false "edited" rate by tile count;
  - the published results for context.
- `notebooks/08_localization_diagnosis.ipynb` (L4, validation only) and `notebooks/09_cp4_gate.ipynb` (L4, P1 dev; P2 behind `P2_ENABLED = False`).
- **CI:** dry runs of notebooks 08 and 09.

Tests:
- `tests/test_shared_trunk.py` (4):
  - the probe's class token and patch tokens equal a plain `CLIPVisionModel` with the same weights, exactly (`torch.equal`), with LoRA made active;
  - this holds at LoRA layers 2–3 and 1–3 (the boundary moves with the fix);
  - the model's tokens are unchanged by the joint pass;
  - the adapters and `requires_grad` are restored afterwards;
  - `forward(probe=True)` keeps the model outputs.
- `tests/test_cp4_eval.py` (11):
  - cross-image batching by tile shape, with the stitched map reproducing the input;
  - pixel counts and pooled metrics checked against hand-worked values, including the held-out-editor split;
  - boundary F1 and histogram AUC;
  - aligned masks reach a ceiling of 1.0, and a 7 px edit vanishes on the 16-grid;
  - edit-size and tile-count bins, and the summaries;
  - the probes separate synthetic features, the L2 is chosen on validation, and the patch-label half rule holds;
  - harmonized P1 compresses a 128² generated PNG and leaves real JPEGs untouched;
  - P2 masks and editor values, and staging is idempotent;
  - gate PASS, FAIL (below 0.721, or losing on harmonized files), red flag and `PENDING_P2`;
  - AP and false-edited bins.

Sandbox runs:
- `ruff check` and `ruff format --check`: clean.
- `python -m pytest -p no:warnings`: 393 passed (after rebasing on main with PR #31).
- `python -m eviground.utils.repo_audit`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 05, 06 (both profiles), 07, 08 and 09: passed.
  - In 08 and 09, a tiny trained model is scored on the fake world.
  - The fake world has no edited validation image, so in 08's dry run only, training images stand in for them.
- **Real-CLIP CPU smoke run** (pinned commit `32bd6428…`, weights hash checked; not in CI):
  - with LoRA made active, the shared pass's probe features equal a plain `CLIPVisionModel` forward exactly (maximum difference 0.0, class token and patches);
  - the model's tokens differ from the plain CLIP's;
  - the tiled path with the real CLIP and a probe head ran on a 300 × 520 image (6 tiles).

Not run here:
- the L4 throughput, the real probe fits and the real metrics (notebooks 08 and 09);
- the DiffSeg30k paper's definitions of binary mIoU, image accuracy and boundary F1. arxiv.org is blocked from this sandbox, so ours are used until they can be checked.

## CP4 PR A — full-coverage tiling, development split, BR-Gen tools — 2026-11-16

From the CP4 plan approved on 16 Nov.

Code:
- `data/tiles.py`: full coverage by default.
  - Stride 256, the last tile aligned to the border, no 3-per-axis cap: a 1024² image takes 16 tiles.
  - The earlier capped rule stays available through `max_per_axis` for comparison.
  - New `tile_count`.
  - Overlapping tiles: mean of the probabilities (unchanged).
- `models/eviground.py`:
  - `predict_tiled` runs fixed-size batches (`TILE_BATCH` = 32) and reports the tile count and coverage.
  - The image-level rule moved into `aggregate_tiles` and `decide`. The rule itself is unchanged: P(real) and P(generated) are means over tiles, P(edited) is the maximum.
- `eval/devsplit.py`: the CP4 development hold-out.
  - P1: 20% of each GenImage validation folder. P2: 20% per base type.
  - Seed 42, salt `cp4_dev_v1`; half rounds up.
  - Identical bytes or pixels form one group and land on one side.
  - The split is frozen by the SHA-256 of its sorted uid list. `load` checks that hash, and `verify` rebuilds the split from the manifest.
- `data/brgen.py`:
  - a parser for `7z l -slt` output;
  - listing of the split archive through its last volume;
  - a summary per folder and kind;
  - image–mask pairing;
  - a seeded stratified subset frozen by its list SHA-256;
  - extraction of named members only.
- `notebooks/07_cp4_prep.ipynb` (CPU):
  - freezes the dev split once, then reuses and verifies it;
  - lists BR-Gen;
  - behind flags, after the subset design is approved: freezes the subset and runs the sample check (untrained model, no scoring).

Tests:
- `tests/test_models.py` (tiling):
  - full coverage on sizes 128–2304;
  - 1024 → 16 tiles; 1000 → last tile at 744;
  - the mean where tiles overlap;
  - stitching identity tiles gives the image back;
  - the capped rule still available;
  - fixed batch sizes (16 tiles at batch 5 → 5, 5, 5, 1).
- `tests/test_cp4_prep.py` (7):
  - the dev split takes 20% per stratum, is deterministic and independent of row order, and depends on the salt;
  - duplicates stay together across strata;
  - save, load and verify, including a tampered list;
  - the BR-Gen listing: parse, summary and pairing (folder and suffix forms);
  - a deterministic subset;
  - a real split zip made with `zip -s` (3 volumes), listed and partly extracted with 7z (skipped where 7z or zip is missing).

Sandbox runs:
- `ruff check` and `ruff format --check`: clean.
- `python -m pytest -p no:warnings`: 377 passed.
- `python -m eviground.utils.repo_audit`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 05 and 07: passed.
  - Notebook 07 ran on the fake world, with a fake BR-Gen zip of 32 files. The subset was frozen and the sample check passed on 4 images, up to 12 tiles.
  - The fake world has only 2 images per GenImage folder, so its dev split is nearly empty; the real folders have 6,000–8,000.

Review fixes (PR #31):
- **Codex (P1):** BR-Gen's README lays its data out as `Au/` (authentic), `Tp/` (tampered) and `Gt/` (masks named `<tampered stem>_mask.png`).
  - The pairing dropped `Gt` but kept `Tp`, so it would have paired nothing.
  - Folders are now matched by exact name for images and masks, plus the `_mask` suffix. `gt` no longer matches inside other words.
  - Authentic images are counted separately.
  - Strata now combine folders with the file's method prefix (`MAT_Stuff_...` → `MAT`).
  - The fake listing and the dry-run archive use this layout.
- **SonarCloud:**
  - a float equality check in notebook 07's sample check is replaced with a tolerance;
  - `devsplit._groups` is split, with a small union-find;
  - a combined test assertion is split.

Not run here: the real manifest (Drive), the real BR-Gen archive (its listing is notebook 07's first job), Colab timings.

## CP3 — training health, seed 42 (Milestone 3) — 2026-11-15

Built from the CP3 plan Dhanush approved on 15 Nov (criteria S1–S3, A1–A4, R1, R2).

Code:
- `eval/metrics.py`:
  - `mask_scores` adds `mask_miou_binary` (the DiffSeg30k paper's binary mIoU over the edited
    crops: the mean of edited-class and background IoU from the pooled confusion matrix),
    `gen_any_pred` and `gen_pred_area`;
  - `per_source_mask_report` gives, per source and class, the share called edited, `any_pred` and
    `pred_area`;
  - the detection selection adds `det_acc` (1 − P(real) ≥ 0.5);
  - `plateau_gain` (best of the last 4 epochs minus the best before).
- `train/trainer.py`:
  - the A1 overfit runs until accuracy ≥ 0.99 and per-image mask F1 on the edited images ≥ 0.80,
    or 1,000 steps. It records accuracy, loss and mask F1 every 25 steps;
  - a non-finite loss term or gradient norm stops the run with no optimizer step. It writes
    `nonfinite.json` and `last.pt`;
  - `grad_norm` in `steps.csv`;
  - a fixed 2,000-image training subset is scored each epoch (`train_eval_*` columns);
  - per-source mask columns;
  - `best_epoch`;
  - a tqdm progress bar per epoch;
  - `summary()` adds `plateau_gain` and `still_improving_at_end`.
- `eval/cp3.py` scores a run folder at its best epoch against the thresholds. It writes them into
  its JSON.
- `notebooks/06_cp3_training.ipynb` (L4), one run per profile:
  - stage the shards;
  - A1 (unified; reused if recorded for the same index; training does not start if it fails);
  - training in the foreground, resumable;
  - curves;
  - `runs/checkpoints/cp3_training_health_<profile>_seed42.json`.
- `configs/base.yaml`: the A1 settings and `train_eval_images: 2000`. The channels_last comment now
  says it stays off.

Tests (`tests/test_cp3.py`, 20, plus the adjusted overfit test in `tests/test_trainer.py`):
- binary mIoU, `gen_*` and real-crop numbers on masks worked out by hand;
- the per-source mask report;
- `det_acc`;
- `plateau_gain`;
- a run writes the new columns and finite `grad_norm`;
- the overfit stops once both targets hold and fails A1 when accuracy arrives after its window;
- a NaN loss at step 3 stops with global step 2, `nonfinite.json` and `last.pt`;
- `still_improving_at_end`;
- `cp3.py` on hand-made `metrics.csv` files:
  - a healthy run passes;
  - each threshold fails on its own (S2a, A2 any-pixel and area, A3, A4);
  - an all-empty-mask model passes A2 and A3 but fails A4;
  - stop-and-ask on a per-class AUC below 0.80, on an R1 gap of 0.17, and on a still-rising run;
  - S1 fails on a NaN step, and A1 fails when the mask target is missed;
  - the detection run and S3;
  - the report round trip.

Sandbox runs:
- `ruff check src tests scripts` and `ruff format --check src tests scripts`: clean.
- `python -m pytest -p no:warnings`: 366 passed (after the review fixes below).
- `python -m eviground.utils.repo_audit`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake`:
  - notebooks 00–05: 8 passed;
  - notebook 06 with `EVIGROUND_PROFILE=unified` and with `detection`: passed.
  - In the dry run the CP3 verdict is FAIL with 2 epochs of a tiny CPU model, as expected.
  - The fake world has no DiffSeg30k real validation image, so R1's gap is empty there. The real
    data has 486 (CP1 run 5).
- Real-CLIP CPU smoke run of notebook 06's code path (not in CI). It used the pinned commit
  `32bd6428…`, with the weights hash checked on load, on the fake-world shards at batch 4:
  - A1 for 4 steps;
  - one training epoch of 3 steps (about 2.4 s per step on CPU);
  - validation with the training-subset evaluation;
  - `cp3.evaluate_run`.
  - Everything ran; 56 s in all.

Review fixes (PR #28):
- Codex: under bf16 and fp32 a non-finite gradient norm stopped the run only after the optimizer
  and the scheduler had moved, so `last.pt` could hold the bad step's weights. The norm is now
  checked before the step. Under fp16 the GradScaler skips such a step itself.
  - New test: a NaN norm at step 3 leaves 2 optimizer calls and `last.pt` at step 2.
  - The test fails with the guard removed and passes with it.
- SonarCloud:
  - `_train_epoch` and `overfit()` split into smaller helpers;
  - a `LAST` constant replaces the repeated `"last.pt"`;
  - `cp3._verdict` replaces a nested conditional;
  - the composite test assertions are split.

Not run here: the L4 times, real convergence, the compute units used, and whether the criteria
hold on the real data. Notebook 06 records these.

## Local copy of the shards — 2026-11-15

The fourth CP2 attempt stalled for over 40 minutes in notebook 05's Cell 4 while it planned
epoch 0. The plan reads about 9,000 edit masks from the shards on Drive, and each read through
Colab's Drive mount took about a quarter of a second. In the sandbox, the computation per edited
draw (decode a 1-bit 1024 × 1024 mask, resize with p 0.3, `edited_crop`) takes 2.8 ms, so 8,000
draws need 0.4 min of compute. The rest was waiting on Drive. Training would have had the same
problem: every crop reads its image from the shards.

Changes (approved by Dhanush, 15 Nov):
- `pack.stage_shards(shard_dir, local_dir)` copies the 18 shards, `index.parquet` and
  `_DONE.json` byte for byte to a local disk:
  - each shard's size is checked against its side-car;
  - the index must hash the same, so the data versions in checkpoints do not change;
  - shards already there with the right size are skipped;
  - it stops if the local disk lacks the space plus 2 GB.
- Notebook 05 stages the shards to `<local data>/shards` (`/content/data/shards` on Colab) in Cell 3
  and reads them there. It records the staging summary in `cp2_model_build.json`.
- `scripts/train.py --stage-local` does the same for training runs; the folder comes from the
  environment, not the command line (SonarCloud S8707).
- `ShardReader` opens each shard once per process and reads with `os.pread`, which takes the
  offset as an argument and so is safe for forked loader workers. It does not pickle descriptors.

Tests (6 new in `tests/test_data_pipeline.py`):
- a staged copy is byte-identical (every shard's sha256, index and `_DONE.json`);
- staging again skips everything, and a cut-off local shard is copied again;
- a disk without space stops staging;
- training items from the local copy equal those from the original (plans, images, masks);
- the reader opens each shard once over two passes of every image and returns the same bytes as
  `pack.read_image_bytes`; a pickled reader carries no descriptor and still reads.
- (Codex) a local copy made for another pack is never reused: after a re-pack whose shard keeps
  its size but not its bytes, every shard is copied again. `_STAGED.json` names the index the
  local shards belong to;
- an interrupted staging resumes for the same pack: shards copied before the interruption are
  skipped, and the marker turns complete at the end.

Sandbox runs:
- `ruff check src tests scripts`: clean.
- `python -m pytest -q`: 345 passed.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 00–05: passed.
- Repo audit: ok.

Not measurable here: Drive's latency, and the copy speed and planning time on Colab. Notebook 05
records both.

## CP2 set-up fixes and the pinned CLIP weights — 2026-11-15

After the first CP2 run on Colab (notebook 05) failed twice:
- The Hugging Face cache on Google Drive held a 79-byte `model.safetensors`: Drive stores the
  cache's symbolic links as text. Notebook 05 now caches CLIP on the local disk.
- Colab's `torchao` 0.10.0 made `peft` raise an ImportError while adding the LoRA layers. The
  setup cell removes a `torchao` older than 0.16.
- `configs/base.yaml` pins the commit (`revision` `32bd6428…`) and the SHA-256 of its
  `model.safetensors` (`weights_sha256` `a2bf730a…`, 1,710,540,580 bytes, from the Hub's metadata).
  `semantic.verified_snapshot` hashes the cached file and loads from that same snapshot folder, so
  the file checked is the file loaded. A different file stops the run (`WeightsMismatch`) with its
  size and hash. The trainer refuses a real backbone without both pins.

Sandbox runs with the real weights (Hugging Face reachable since 15 Nov; scratch scripts, not in
CI), on CPU:
- **Weights:** the downloaded file is 1,710,540,580 bytes and its SHA-256 equals the Hub's
  `a2bf730a…`.
- **Model through `build_model(base)`:**
  - 322,706,209 parameters, of which 303,179,776 are frozen CLIP and 19,526,433 trainable;
  - M2 trainable 655,872 (LoRA 131,072, projection 524,800);
  - M3 3,653,532; M4 fusion 13,138,945; heads 2,078,084.
- **Shapes at 256 (batch 2):** logits (2, 3), mask (2, 1, 256, 256), α (2, 256), g (2, 1),
  Z (2, 256, 512). α sums to 1; g in (0, 1).
- **No-grad split:** layers 0–19 without autograd give exactly CLIP's own last hidden state
  (max abs difference 0.0).
- **One AdamW step:**
  - the frozen CLIP SHA-256 is unchanged (`cdd5e490…`), no frozen parameter gets a gradient, and
    LoRA moves;
  - step 1 gives non-zero gradients to the 8 `lora_B` tensors only (B starts at zero); from step 2
    all 16 LoRA tensors have gradients;
  - loss terms on a random batch: total 2.558, CE 1.182, BCE 0.492, Dice 0.796, attention 0.877.
- **CPU times** (not the L4's): forward 1.24 s and a training step 1.90 s for batch 2.
- **The hash check:**
  - the verified load passes, with the same frozen-CLIP hash as above; hashing the file takes
    1.4 s;
  - a Drive-like 79-byte file stops (`WeightsMismatch`), and so does the file one byte short.

Tests (`tests/test_clip_weights.py`, 6, CI, no network; a fake cache with a tiny CLIP laid out
like Hugging Face's):
- matching weights load from the checked snapshot folder, with identical tensors;
- a wrong hash stops before anything loads;
- the 79-byte link text is caught;
- the file is hashed once per version in a process;
- the config pins both values;
- the trainer refuses a real backbone without `weights_sha256`.

## Milestone 3, PR C — trainer, checkpoints, notebooks 04 and 05 — 2026-11-15

Modules: `train/checkpoint.py`, `train/trainer.py`, `scripts/train.py`; metrics added to
`eval/metrics.py`; `configs/base.yaml` gets `train` and `model.semantic.revision`.

With Dhanush's five fixes of 14 Nov 2026:
1. **Checkpoint contents:** the full `state_dict` of every module except the frozen CLIP, buffers
   included (BatchNorm running statistics, `num_batches_tracked`); optimizer, scheduler and scaler
   states; Python, NumPy, torch CPU and CUDA RNG states.
2. **Frozen CLIP:** the Hugging Face revision is pinned (`model.semantic.revision`; training refuses
   a real backbone without one). Its SHA-256 over the frozen weights is computed once per run and
   stored in every checkpoint; a resume checks both and stops on a mismatch.
3. **Augmentation RNG:** keyed by (seed, epoch, item) only, so items are identical with 0 or 2
   loader workers and after a restart.
4. **Validation metrics:** the selection score uses the mean of per-image mask F1 on edited crops.
   Also logged: pooled F1 and IoU, and on real crops the share with any predicted edited pixel and
   the mean predicted area. Losses and metrics are computed in float32 outside autocast.
5. **Versions:** the shard index sha256, the manifest sha256 and the mask-rule version are in every
   checkpoint and every `metrics.csv` row; a resume on a different index stops.

Rest of the trainer:
- AdamW with two learning rates; 1-epoch linear warm-up, then cosine; gradient clip 1.0;
  gradient accumulation.
- bf16, or fp16 with GradScaler.
- Early stopping, patience 4.
- `step_*.pt` and `last.pt` every 500 optimizer steps and at epoch end (2 step files kept);
  `best.pt` on improvement; atomic writes.
- `metrics.csv`, `steps.csv`, and `compute.json` (cumulative GPU hours × CU rate).
- 64-image overfit mode: fixed crops, augmentation and dropout off, base learning rates.
- `scripts/train.py`: `--resume auto`, `--fresh` (moves the old folder aside), `--dry-run`,
  `--max-steps`, `--overfit`, `--kill-at-step` (ends the process abruptly, for CP2).

Notebooks:
- `04_pack_shards.ipynb` (CPU): Drive free space, then stage, pack (stops and asks if free space <
  estimate + 10 GB; resumable) and verify. Writes `runs/checkpoints/pack_shards.json`.
- `05_cp2_model_build.ipynb` (L4): parameter counts, shapes, frozen-CLIP SHA-256 before and after
  training steps, 64-image overfit, kill and resume in subprocesses (reference run vs a run killed
  after step 30 with checkpoints every 20, compared step by step), timings and peak memory with and
  without channels_last, loader images/s, and a validation pass. It resolves and records the CLIP
  commit hash. Writes `runs/checkpoints/cp2_model_build.json` with the script's verdict.

Tests (`tests/test_trainer.py`, 17, CPU, tiny CLIP on the fake-world shards):
- a 2-epoch run writes `config.yaml`, `metrics.csv` (with every new column), `steps.csv`,
  `compute.json`, `best.pt` and `last.pt`, and keeps 2 step files;
- a checkpoint holds BatchNorm buffers, LoRA, no frozen CLIP key, the RNG states and the versions;
- a model loaded from a checkpoint gives bit-identical eval outputs (logits, mask logits, α, g);
- kill after step 3 (checkpoint at 2), then `resume auto`: global step 2, then the same batches
  and the same losses (within 1%) as the uninterrupted run up to step 12;
- an interrupted save keeps the previous checkpoint;
- rotation;
- a changed frozen weight or a different revision stops a resume; so does a different shard index;
- a real backbone without a pinned revision is refused;
- items are equal with 0 and 2 workers and in a restarted process;
- the overfit path runs with dropout 0;
- binary AUC equals the pair count;
- per-image vs pooled mask F1, IoU and the real-crop numbers;
- the selection score for both profiles.
- review fixes: a short last accumulation group is scaled by its own size; a resume with another
  profile or seed stops; an epoch stopped by `max_steps` or a kill reports its loss over all its
  micro-batches; checkpoints load with `torch.load(weights_only=True)`.

Sandbox runs:
- `ruff check src tests scripts` and `ruff format --check`: clean.
- `python -m pytest -q`: 333 passed (17 new, after the SonarCloud and Codex review fixes).
- `python -m eviground.utils.repo_audit`: ok.
- `EVIGROUND_DRY_RUN=1 pytest --nbmake` on notebooks 00–05: passed. The dry run of 05 runs every
  cell on the CPU with the tiny CLIP: kill and resume passes there. The overfit and timing checks
  report FAIL in the dry run, as expected: 4 CPU steps and no GPU, so nothing is measured.

Not run here:
- the real CLIP (Hugging Face is blocked from the sandbox, so the commit hash could not be looked
  up: notebook 05 resolves and records it, and it then goes into `configs/base.yaml`);
- GPU timing, memory, bf16 and the real overfit (notebook 05 on an L4);
- packing the real data (notebook 04).

## Milestone 3, PR B — packed shards, datasets, augmentation — 2026-11-15

Modules (decisions of 15 Nov 2026): `data/pack.py` writes the training and validation images that
harmonization keeps as their **harmonized bytes** (`cp1.harmonized_bytes`: the CP1 plan, COCO
prepared to its DiffSeg30k size) into ~1 GB tar shards per source, plus a **1-bit PNG edit mask**
for every edited image (CP1 reading rule: an AI base's value read as 0); `index.parquet` holds one
row per image (labels, roles, generator / editors, byte offsets of image and mask, `mask_base`,
`valid_crop`, the mask-rule version); `_DONE.json` holds the counts, the manifest sha256 and the
size estimate. Before writing, the shard size is estimated from a seeded sample per source and
packing stops (`PackSpaceError`, stop and ask) if Drive free space < estimate + 10 GB. Shards are
written atomically with a side-car index, so an interrupted pack resumes; with `_DONE.json` for the
same manifest nothing is read again. `data/datasets.py` (`ForensicDataset`) reads by offset and
decodes (no harmonization at load time, no local cache): training follows a deterministic plan per
(seed, epoch) with the sampler shares, `edited_crop` on the (resized) edit mask and replacement of
failed draws (`epoch.epoch_plan`); edited images with `valid_crop` False are left out and listed;
validation has one fixed crop per image from (seed, uid), no augmentation; the detection profile
has GenImage real and generated only. `data/transforms.py` (`Augment`): resize 0.5–1.5× before
the crop (p 0.3, short side ≥ 256, mask nearest), flip (p 0.5), JPEG 60–100 (p 0.5), blur σ 0–2
(p 0.2), noise σ 0–3/255 (p 0.2); `Augment(enabled=False)` switches it off. `epoch.CropFn` may now
return any crop spec (here box + resize factor).

Tests (`tests/test_data_pipeline.py`, 19, CPU, ~13 s, on the fake CP1 world): the shards hold
exactly the kept training and validation images, no test or excluded row; no held-out uid and no
Glide/Kolors name; packed bytes equal a fresh harmonization for every packed image of every source
(coco, diffseg30k, genimage, sidset); shard round trip by offset; the stored edit mask equals
`edit_mask(mask, mask_base) > 0`, 1-bit, with the rule version; the free-space stop (also before
anything is written); no source image read when `_DONE` exists; a shard cut off mid-write is
rewritten and the rest skipped; training crops are 256 × 256 inside the (resized) image, never
padded; edited crops hold 1–90% edited pixels; class ratio exactly 1 : 1 : 1 and every source
share within ±5% over 3,000 draws; plans and items identical across instances (resume at any
item) and different between epochs; never-valid edited images left out and listed; validation
crops fixed across instances and epochs; the detection profile; flip and resize keep image and
mask aligned (> 98% agreement on a test pattern); augmentation off returns the crop unchanged;
photometric steps are seeded.

Sandbox runs: `ruff check src tests`, `ruff format --check src tests`, `python -m pytest -q` →
313 passed. Fake-world pack: 73 images, 21 shards, estimate 3.29 MB vs 3.33 MB written,
`verify_sample` 73 / 73 equal. Not run here: packing the real data on Drive (notebook 04, PR C),
loader throughput (CP2 notebook).

## Milestone 3, PR A — model M1–M4 and losses — 2026-11-15

Modules (spec section 8; brief M1–M4): `models/views.py` (RGB 224 CLIP-normalised; SRM with the
3 fixed kernels, clamped ±3 on the 0–255 scale, scaled to ±1; orthonormal 8 × 8 block DCT, log,
per-image standardised, at full resolution; computed on the model's device), `models/semantic.py`
(`transformers` `CLIPVisionModel`, SDPA, frozen, `peft` LoRA r 8 / α 16 on q_proj and v_proj of
layers 20–23, projection to d = 512; layers before the first LoRA layer run without autograd; works
with transformers 4.x and 5.x), `models/trace.py` (`timm` efficientnet_b0, 4 input channels,
strides 4/8/16, 1 × 1 projection; optional channels_last), `models/fusion.py` (2 bidirectional
pre-norm cross-attention blocks, gate g, Z), `models/heads.py` (attention pooling + MLP → 3 logits
with α; FPN-lite mask decoder), `models/eviground.py` (`EviGround`, `build_model(cfg.model)`,
`param_groups` for the two learning rates, `predict_tiled`, `disable_dropout`), `data/tiles.py`
(test-time tiles) and `train/losses.py` (CE with label smoothing; BCE on all samples, Dice on
edited samples only; L_att on edited samples). Configs: `model`, `loss`, `optim` sections in
`configs/base.yaml`; `configs/debug.yaml` uses a tiny random CLIP (no download). CI now installs
`torch` 2.14.1 and `torchvision` 0.29.2 together from the PyTorch CPU index (the first CI run
failed because `timm` pulled a CUDA `torchvision` from PyPI: `operator torchvision::nms does not
exist`), and `timm` 1.0.31, `peft` 0.21.3, `transformers` 5.20.2, all pinned and binary-only; the
unused `open_clip_torch` dependency is removed.

Tests (`tests/test_models.py`, 23, CPU, ~8 s): view shapes at full resolution; the DCT matrix is
orthonormal and a constant block has only the DC term; SRM of a constant image is 0; views are
deterministic; frequency view for sides that are not multiples of 8; only LoRA and the projection
train in M2; M2 trainable parameters at real ViT-L/14 width (built on the meta device, no weights)
0.66 M < 5 M, LoRA 131,072; the no-grad split gives the same tokens as the full CLIP; output shapes
at 128 / 256 / 512 with α summing to 1 and g in (0, 1); only the planned parameters require grad;
frozen CLIP weights byte-identical after an AdamW step while LoRA moves; gradients reach the
trace branch's first convolution, the class head and the decoder; the two parameter groups and
their learning rates; dropout off makes the model deterministic; Dice 0 for identical masks; a
real image feeds BCE (non-zero gradient) and not Dice; L_att 0 inside the mask and 1 outside; the
loss is the weighted sum of its terms; the tile rule (≤ 9 tiles, never padded, full coverage up
to 768 px per side, 75% for 1024); stitching; tiled prediction with one edited tile returns
"edited" and a mask of the image size; channels_last in the trace branch gives the same output.

Sandbox runs: `ruff check src tests`, `ruff format --check`; `python -m pytest -q` → 294 passed
(torch 2.15, transformers 5.23); `tests/test_models.py` again in a venv with the CI pins
(transformers 5.20.2, timm 1.0.31, peft 0.21.3) → 23 passed. Not run here: the PyTorch CPU index
is blocked from the sandbox, so the exact torch 2.14.1 + torchvision 0.29.2 CPU pair is checked
by CI only.

Not testable here: the real CLIP weights, GPU timing and memory (CP2 notebook on an L4).

Open question for Dhanush (Milestone 5): for a side over 768 px, at most 3 tiles per axis cannot
cover the image (1024 → 75% per axis); uncovered pixels currently take the nearest tile's value.

## Milestone 2 — DiffSeg30k mask reading, strict hold-out, real negatives back — 2026-11-14

Changes (data/README decisions 15–17): `diffseg_mask.read_base` / `edit_mask` (an AI base's value,
the border-majority and most frequent value, read as 0); `splits.diffseg_role` with edit values
and the reading (strict Glide/Kolors hold-out; P2 = Glide/Kolors as editor; `real_pool`,
`excluded_base_unclear`, `excluded_no_edit`, `excluded_holdout_base_only`); `cp1.diffseg_bases`
(a watched pass over the masks: base, reading, valid crop on the read mask);
`cp1.base_reading_report` and the stop flag `diffseg_base_unclear_over_5pct`; `cp1.draw_crops`
(training crop rule on read masks, failed edited draws replaced round by round, failure kinds
and the share histogram of every failed try); sampler real = 1/3, 1/6, 1/4, 1/4; manifest columns
`mask_base`, `base_type`, `base_reading`, `edit_values`, `valid_crop`. The fake DiffSeg30k masks now
have the real layout (real bases, AI bases labelled on every pixel, a Glide base, a Kolors base in
validation, an AI base without an edit, a border/count disagreement, empty masks).

Tests: `tests/test_diffseg_mask.py` (AI base agree, real base, unedited, no edit, disagreement,
border counts, the report's counts and disagreement share, `draw_crops` keeps 40 crops per class
when half the edited images never crop, histogram = 10 tries per failure, replacements logged);
`tests/test_cp1_core.py` (the strict hold-out role matrix, 14 cases; quotas and draws with the new
real shares; q75 = 1/3 per class); the fake CP1 run (roles, base readings 12/2/2/12/2, P2 6 = 4 real
+ 2 AI base, valid crops 6 of 8, 60 edited crops after replacement, real crops 20/10/15/15).

After Codex review of PR 18: the record also holds the share histogram of every failed try
(replacement rounds included) beside the first-pass one, and two stop flags,
`diffseg_masks_unreadable` (a mask that could not be read again in the base pass) and
`edited_crops_unresolved` (edited crops still failing after every round); tests in
`tests/test_diffseg_mask.py`.

Not testable here: the real masks and CP1 run 5 (Colab).

## Milestone 2 — DiffSeg30k mask-label diagnostic — 2026-11-14

Why: notebook 02 (run 4 reproduced) found 1,583 of 1,585 failed edited draws above 90% and 5,930
of 11,382 edited training images without a valid window; their masks have no zero pixel.
`mask_diagnostic` + `notebooks/03_mask_diagnostic.ipynb` (read-only, in the CI dry runs) measure
every DiffSeg30k mask: value shares, full vs partial by size and role, dominant and second value,
per value dominant vs minority, the Glide / Kolors hold-out as background vs edit region, and a
counted-only what-if (a full mask's dominant value as background: valid crops, masks left without
an edit, role changes); `mask_examples.png` shows 8 full and 4 partial masks.

Tests (`tests/test_mask_diagnostic.py`): mask facts of a full two-value mask, a partial mask and an
empty mask; the what-if edited pixels; on six synthetic masks (full 7+3, full Kolors base + 5, full
single value, partial Kolors edit in P2, partial, empty): counts, dominant values, per-value
split, Kolors as background in the hold-out vs as an edit in P2, one mask with no edit left, the
role changes (hold-out → training pool, training pool → unedited), valid crops now 1 → what-if 3,
and the example sheet written. The notebook dry run uses the fake world, which has no full masks
(they are covered by the unit tests only).

Not testable here: the real masks (Colab).

## Milestone 2 — training crop rule, crop diagnostic, per-source accuracy — 2026-11-14

Changes: `crops.edited_crop` (random edited pixel, window uniform among those containing it,
1–90% share or an optional pixel minimum, 10 tries), `crops.window_counts` and
`crops.has_valid_window` (every window, integral image); `epoch.epoch_plan` (sampler draws, a
failed crop replaced from the same pool, failures and replacements logged) and
`epoch.screen_never_valid`; `crop_diagnostic` and `notebooks/02_crop_diagnostic.ipynb` (CP1 run 4's
edited draws rebuilt from its manifest, every try's share recorded; added to the CI notebook
dry runs); `eval.metrics.per_source_accuracy` and `metrics_row` (validation accuracy per source and
class, flat columns for `metrics.csv`). CP1 keeps `crops.edited_box`.

Tests (`tests/test_crops_epoch.py`, `tests/test_metrics.py`, one in `tests/test_cp1.py`): over 4,000
draws the edited blob's position inside the crop is uniform (mean 128 ± 6, sd 66–82, every quarter
> 850 of 4,000, < 10% in the central 64 × 64; a centred crop would give 100%); windows stay inside
the image at a border; every returned box has a 1–90% share; a 300-pixel edit fails the 1% rule
and passes a 256-pixel minimum; empty and small masks fail cleanly; `has_valid_window` equals brute
force on 40 random masks; replacement keeps the exact per-pool quotas when half the edited images
never crop, logs the failures, is deterministic, and stops when no edited image can crop; the
tracing twin repeats `edited_box` exactly (same random numbers, both outcomes covered); the
diagnostic sorts failed draws into below-1% and above-90% kinds, its histogram sums to 10 tries per
failure, and on the fake CP1 run it rebuilds the run's edited draws (60) and reproduces its
failure count; per-source accuracy on a hand-computed case, a missing source, unknown sources,
fixed column order.

Speed: `edited_crop` sums only the chosen window (0.13 ms per call on a 1024 × 1024 mask; a
whole-image integral image per call took about 27 ms).

Not testable here: the real failure histogram (Dhanush runs notebook 02 in Colab), and training.

## Milestone 2 — spec review: DiffSeg30k unedited images out, q75 thirds by source — 2026-11-14

Changes (data/README decisions 13–14): `splits.diffseg_role` gives empty-mask train images
`excluded_unedited`; `harmonize.plan_image` compresses the seeded third (`q75_third`) of every
DiffSeg30k and COCO image at quality 75, 4:2:0 whatever the class; `sampler.SHARES` real =
GenImage 1/3, SID-Set 1/6, COCO 1/2; `cp1.jsonl_census` counts the keys of every `.jsonl` line
(stop flags `diffseg_jsonl_unexpected_keys`, and `diffseg_jsonl_census_incomplete` when no
`.jsonl` is found or a line cannot be read, after Codex review of PR 15); `diffseg_mask_report.size_indication`;
`blockiness_check` adds "real (coco) vs diffseg30k edited" and reports the within-DiffSeg30k pair
as not applicable; `run_cp1` records `quality_75` per class and per class × source and the failed
edited crop draws; the fake DiffSeg30k `.jsonl` now has the real layout (two file names per line).

Tests: role of empty masks; the same q75 ids in every class of DiffSeg30k and COCO, GenImage
generated never; q75 is exactly 1/3 of every class from the sampler shares; quotas and draws with
the new shares; `.jsonl` census (line, unparsable and key counts; an unexpected key is reported);
size kinds; pooled real vs DiffSeg30k edited under the stop rule; q75 share per class; the fake run
records census, size indication, q75 shares, new separation and failed crop draws; the pooled
blockiness test rewritten for the new mix (no stop when every class is one third q75; a real
SID-Set difference still stops). Results of the commands are in the pull request.

Not testable here: the full `.jsonl` census and every run-4 number (real data, Colab).

## Milestone 2 — CP1 run-2 decisions: q75 third for edited, accepted GenImage trace — 2026-11-14

CP1 run 2: criteria met; blockiness stop (GenImage 0.6675, a generator trace by the raw and
mid-block diagnostics; pooled real/generated vs edited 0.6205 / 0.642 from SID-Set's q75 images).
Decisions implemented: `harmonize.q75_third` sends a seeded third of DiffSeg30k edited images to
quality 75, 4:2:0; `CP1Config.blockiness_accepted` lists GenImage generated vs real, which
`blockiness_check` reports under `accepted` and leaves out of the stop flag.

| Check | Result |
|-------|--------|
| `ruff check src tests` / `ruff format --check src tests` | All checks passed / 42 files already formatted |
| `pytest` | 231 passed (3 new) |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |

## Milestone 2 — CP1 fixes after the first real run — 2026-11-14

CP1 run of 14 Nov (FAIL): 169 cross-split duplicate groups (ImageNet photos in both GenImage SD v1.4
training and GenImage Validation), near-duplicates in P2 (1.18%) and P3b (5.88%), blockiness
separations 0.6675 (GenImage) and 0.6587 (SID-Set); 5,000 DiffSeg30k training images with an empty
mask. Decisions by Dhanush, implemented: `cp1.filter_training_leaks` leaves training images out when
they equal (file or pixel hash) or nearly equal (pHash ≤ 6) a test image, and leaves SID-Set tampered
out of training; identical images inside the pool share a split; DiffSeg30k empty-mask images become
real negatives; new sampler shares; blockiness diagnostics on raw files and mid-block lines. Also the
Codex finding on PR #12: `stage()` re-extracts missing or resized files of an unpacked copy.

| Check | Result |
|-------|--------|
| `ruff check src tests` / `ruff format --check src tests` | All checks passed / 42 files already formatted |
| `pytest` | 228 passed (tests changed where the decided behaviour changed: duplicates of test images are now left out of training instead of failing the run; 7 new tests) |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |

## Milestone 2 — COCO negatives at DiffSeg30k sizes; exact staged-file check — 2026-11-14

Decisions of 14 Nov 2026 (Dhanush): COCO negatives match DiffSeg30k's measured image sizes (each
negative gets a size drawn, seeded by its id, from the sizes of the DiffSeg30k training images;
Lanczos resize by the smallest factor that covers it, then centre crop; no resampling when the
size already matches); defaults 2–5 confirmed; the Codex finding on PR #11 accepted: `stage()`
records every staged file with its size, and `check_staged` requires the disk to match that list
exactly (missing, other size and extra files each stop CP1). The COCO pHash after preparation
is computed after the scan (its target size is known only then).

| Check | Result |
|-------|--------|
| `ruff check src tests` / `ruff format --check src tests` | All checks passed / 42 files already formatted |
| `pytest` | 223 passed (4 new; 7 tests that edit staged files on purpose now record the edit as the staged state) |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |

## Milestone 2, fix — CP1 started without local copies — 2026-11-14

Colab report: Cell 4 printed `indexed 136000 images: {'genimage_val': 100000, 'sidset': 36000}` and
stopped with `ValueError: no colour JPEG among the measured images`. The local copies of GenImage
SD v1.4, DiffSeg30k and COCO were gone (a new runtime; Cell 3 not run again), so only the datasets
indexed from Drive were found. Changes: `cp1.check_staged` runs first in `run_cp1` and stops with
"local copies are missing or incomplete (…): run Cell 3 first" unless every local copy has its
`_STAGED.json` and at least the files it records; unpacked files take the archive's timestamp and
copied files keep the source's, so staging again reproduces the same size and time and the scan
saved on Drive is reused; the quality-distribution error says how many images were measured.

| Check | Result |
|-------|--------|
| `ruff check src tests` / `ruff format --check src tests` | All checks passed / 42 files already formatted |
| `pytest` | 220 passed (3 new; the class-count mismatch test now changes the card count instead of deleting a staged file) |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |
| `test_scan_cache_survives_restaging` on the previous `storage.py` | fails: `scanned again after re-staging` |

## Milestone 2, fix — CP1 scan stalls on Drive reads — 2026-11-13

Colab report: Cell 4 resumed from the scan cache (COCO, DiffSeg30k, GenImage SD v1.4 done) and printed
three part lines for GenImage Validation (44,000 / 46,000 / 48,000 of 100,000 at 8, 164 and 135
images/s), then nothing while the cell kept running; every run got three parts further. GenImage
Validation is read in place on the Drive mount; one read that never returns blocked the whole
process pool, which had no timeout. Change: `eviground.data.watchdog.run_watched` knows which image
each worker is reading; a worker busy with one image for more than 180 s is killed and replaced and
the image retried; after two stalls the image is recorded as `read timeout`. Progress every 500
images. Any image that could not be read sets the new stop flag `images_not_scanned`.

| Check | Result |
|-------|--------|
| `ruff check src tests` / `ruff format --check src tests` | All checks passed / 42 files already formatted |
| `pytest` | 217 passed (7 new: 6 in `test_watchdog.py`, 1 in `test_cp1.py`) |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |
| Old `scan()` (main before this fix) with one read that never returns, `timeout 60` | still stuck after 60 s (exit 124) |
| New scan, same situation (`test_stuck_drive_read_is_recorded_and_run_finishes`) | finishes in about 11 s; the image is a `read timeout`, run reports `images_not_scanned` |

Not testable here: why the Drive mount stalls, and whether a retried read succeeds on Colab.

## Milestone 2, fix — staging speed and progress — 2026-11-13

Colab report (13 Nov, after the merge of PR #8): Cell 3 of `01_cp1_data_integrity.ipynb` was still
running after 3.25 h with no output. Cause: `copy_files` copied SID-Set's 48,003 loose files one at
a time through the Drive mount (one round trip per file), and neither `copy_files` nor
`extract_archive` reported progress. Change: `copy_files` copies with 16 threads (same `.part` →
rename, same size check, resumable); both functions log a line every 2,000 files or 2 GB (files/s,
MB/s, minutes left); `stage` logs the time per dataset.

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` (all) | 210 passed (5 new) |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |
| Copy of 200 files with 30 ms simulated latency per file | 1 worker 6.12 s, 4 workers 1.54 s, 16 workers 0.41 s |

Not measurable here: the real Drive mount's per-file latency and how many parallel reads it serves.

## Milestone 2, step 4 — checkpoint CP1 code (spec revision 49) — 2026-11-13 — branch `claude/nice-curie-3tzxzr`

New: `hashing.py` (file / pixel SHA-256, 64-bit DCT pHash, chunked Hamming search),
`harmonize.py` (the per-source table of spec section 8), `crops.py`, `sampler.py` (shares of
section 9), `splits.py` (10% stratified split, DiffSeg30k editor map and hold-out), `shortcut.py`
(lookup-table metadata classifier, weighted threshold classifier), `cp1.py` (`run_cp1`), the
`coco_val2017` download step (`url_files`, resumable, CRC-tested), `fake_cp1.py` (a fake Drive with
the layouts the probe found) and `notebooks/01_cp1_data_integrity.ipynb`.

### How to run

```
ruff check src tests && ruff format --check src tests
python -m pytest -q
EVIGROUND_DRY_RUN=1 python -m pytest -q --nbmake notebooks/00_setup_and_data.ipynb \
    notebooks/00c_data_download.ipynb notebooks/00b_data_probe.ipynb \
    notebooks/01_cp1_data_integrity.ipynb
python -m eviground.utils.repo_audit --history
```

### Results (sandbox: Python 3.13.18, 4 CPU cores, no GPU)

| Check | Result |
|-------|--------|
| `ruff check src tests` / `ruff format --check src tests` | All checks passed / 40 files already formatted |
| `pytest` | 205 passed, 0 failed (149 before; 56 new: 36 in `test_cp1_core.py`, 20 in `test_cp1.py`) |
| `pytest` with 8.1 GB free simulated (CI runner) | all passed |
| Notebook dry runs (`00`, `00c`, `00b`, `01`) | 4 passed |
| Repository audit (full history) | ok |

### What the new tests cover

- Harmonization: every row of the spec table (GenImage, SID-Set incl. at-target / below-75 / other
  subsampling, DiffSeg30k and COCO); the written JPEG has exactly the planned quality and
  subsampling; the same image always gets the same draw; compression raises blockiness.
- Hashes: pixel hash equal for the same pixels in different files; pHash near for a recompressed
  or resized copy (<= 2 bits), far for another image (> 10 bits); Hamming search and counts.
- Splits: 10% per stratum, independent of input order; DiffSeg30k roles for every case (training
  editors, Glide/Kolors in train and validation, empty mask, value outside 0-8).
- Sampler: quotas per class and source, seeded draws; crops: 256 without padding, edited share
  1-90% in at most 10 tries.
- Metadata classifier: finds a leaking feature (single-feature and drop-one scores), ~0.5 on noise;
  on the fake world the raw-file positive control scores > 0.9 while the harmonized criterion
  features (one format, one crop size) stay <= 0.60.
- CP1 end to end on the fake Drive: COCO download, CRC failure, wrong member count, interrupted
  download resumed with HTTP Range (also the real `NetworkSources` against a local HTTP server);
  Claims-Edit reservation rules; class counts against the manifests (and a deliberate mismatch);
  a cross-split exact duplicate (FAIL); a recompressed copy of a test image among the COCO
  negatives (dropped); near-duplicates over 1% of a test set (stop); a mask value 9 (stop); a
  corrupt image (excluded, counted); the scan resumed from its cache with no re-scan; the same
  result with 1 or 2 worker processes.
- Pooled blockiness uses the sampler's mix of sources, each measured image once with a weight.
  Found while testing: redrawing values from a small group puts copies in both cross-validation
  folds and made a no-difference case score 0.62; fixed with weights and covered by two tests.

### Not testable here

Real images and their volumes (about 256,000 scanned, about 70 GB staged), Colab's CPU and Drive
(FUSE) speed, the COCO host, real JPEG headers with non-standard tables (quality then reported as
the nearest standard quality, flagged as not exact). Scan cost measured here on one core: 5 ms
(ImageNet JPEG), 11 ms (512² PNG), 42 ms (1024² PNG), 32 ms (COCO photo incl. preparation).

## Milestone 2, fix — complete a partially copied generator (Midjourney) — 2026-11-13

Colab run of 13 Nov (`00c`, `['genimage_val']`): the SD v1.4 fill tar had been left unreadable on
Drive and was refilled from the archive; the count then stopped at 96,411 of 100,000 with
`adm 6000/6000, biggan 6000/6000, glide 6000/6000, midjourney 6000/2411, sd14 6000/6000,
sd15 8000/8000, vqdm 6000/6000, wukong 6000/6000` (real/generated). Midjourney's generated images
were only partly copied by the retired downloader (2,411 of 6,000); with SD v1.5 at 8,000 + 8,000 the
total is 100,000.

Change: whenever the Kaggle Validation archive is listed, its images per generator and class are
saved to `archive_counts.json`; the fill plans only images that are absent from Drive in generators
holding fewer images than the archive (existing files are never overwritten; a different size is
only counted and reported); verification requires Drive to equal the archive counts. Once the counts
are recorded, a complete folder is verified without any download.

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 149 passed, 0 failed, 0 skipped (2 new; both fail on the previous code) |
| Space-related tests with 8.1 GB free (CI runner) | 17 passed |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |

## Milestone 2, fix — real Drive quota, Drive upload cache, tar-to-loose conversion — 2026-11-12

Third Colab download run (15:4x UTC, commit `ad3b07d`, before PR #5): the Validation fill worked
(the tar version: `genimage_val_{glide,sd14,vqdm}.tar`, 12,000 images each) but verification stopped
at `96411 images, expected 100000`; the SD v1.4 subset stopped with "needs 23.1 GB on Drive, only
14.6 GB free". Finding: on Colab, the free space of the Drive mount is the machine's own disk, not
the Drive quota. Evidence from the run records: writing 26.5 GB of DiffSeg30k zips to Drive moved
"Drive free" from 109.6 to 104.8 GB together with the local disk (115.4 → 110.4), and the 96.5 GB SD
v1.4 archive on the local disk brought "Drive free" to 14.6 GB. Fixes:

- Drive space is read from the Google Drive quota through the Drive API whenever the data folder is
  on the Colab mount (`env.drive_quota`); if it cannot be read, the step stops (`SpaceUnknownError`).
- Before each write to Drive, the step waits (up to 30 min) until the local disk can hold the
  Drive upload cache, instead of failing next to a large local archive.
- A failed verification records its measured numbers (`details`), and the Validation error lists
  real/generated counts per generator.
- Fill tar files from that run are unpacked into the loose layout (each file checked against its
  recorded SHA-256), then removed; nothing is downloaded again.
- Notebook `00c` reports the Drive quota (used and free) and the local disk separately.

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 147 passed, 0 failed, 0 skipped (7 new) |
| Space-related tests with free space limited to 8.1 GB (CI runner) | 14 passed |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |

## Milestone 2, step 5 (Validation fill as loose files, not tar) — 2026-11-12

Change of decision (12 Nov 2026): the five generators already on Drive are loose files, so GLIDE,
SD v1.4 and VQDM are now written as loose files in the same layout (the archive's own paths,
`GenImage/<Generator>/val/{ai,nature}`) instead of one tar file each. Each file is written as
`.part` and renamed; `fill_plan.csv` lists every file and size and `fill_manifest_<generator>.json`
its SHA-256. Every run re-checks the filled files' sizes, so a file cut short on Drive is rewritten
(this replaces the damaged-tar repair of PR #4). `stage("genimage_val")` reads everything in place.

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 141 passed, 0 failed, 0 skipped |
| Fill tests with free space limited to 8.1 GB (as on the CI runner) | 6 passed |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |

Fill tests now check: files land in the existing layout, the 20 loose images already there are not
touched, no tar or `.part` file is left, SHA-256 manifest written, archive deleted, second run makes
no network call, nothing to stage; resume after an interrupted download; a generator missing from
the archive is named; space check before any download; a filled file cut short is rewritten.

## Milestone 2, fix — stale code in a reused Colab runtime — 2026-11-12

Second Colab run of `00c_data_download.ipynb` (15:13 UTC, commit `ad3b07d`): both datasets failed in
1.4 min. `genimage_sd14`: `MissingSecretError` (Kaggle secret not visible to the notebook).
`genimage_val`: the same error text as the first run, including the SD v1.5 symptom fixed in PR #3,
and Cell 2 printed the old registry text. Cause: the runtime from the first run was reused, so
Python still held the old `eviground` modules in memory while `git pull` had updated the files.
Fix: Cell 1 of every notebook now drops loaded `eviground` modules after pulling; Cell 2 of `00c`
prints which code file is in use.

| Check | Result |
|-------|--------|
| Module purge (sandbox: patch a loaded module, run Cell 1's lines, import again) | fresh module loaded, `fill_from` present |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |

## Milestone 2, step 4 (fill the missing GenImage generators as tar files) — 2026-11-12

Decision of 12 Nov 2026 ("recommended, tar"): GLIDE, SD v1.4 and VQDM, missing from the Validation
folder on Drive, are taken from the Kaggle `vtphatt2/GenImage-Validation` archive (to the local disk,
not extracted, deleted afterwards) and stored as `genimage_val_<generator>.tar` beside the five loose
generators. Verification counts loose files and tar members together; `stage("genimage_val")`
unpacks the tars, the loose folders are read in place. If a generator is not in the archive, the
step names it and lists the archive's top folders, and no `_DONE` is written.

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 141 passed, 0 failed, 0 skipped (6 new) |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |

New tests: the three generators are added as three tar files and the five loose ones are untouched,
the archive is deleted, a second run makes no network call, staging unpacks the tars; an interrupted
archive download resumes; a generator absent from the archive is named, existing tars are kept, no
`_DONE`; too little space stops before any download; the probe reads a filled Validation as complete;
a fill tar damaged on Drive is rewritten on the next run instead of crashing the count (review of
PR #4). CI runs these tests with a tiny space estimate (the runner has about 8 GB free).

## Milestone 2, fix — GenImage generator names of the Kaggle mirror — 2026-11-12

First Colab run of `00c_data_download.ipynb` (12 Nov 2026): DiffSeg30k downloaded and verified
(SHA-256), SID-Set verified in place, GenImage Validation **failed**. Listing the Drive folder
showed the mirror's layout `Validation/GenImage/<Generator>/val/{ai,nature}` with only 5 of 8
generators (`ADM`, `BigGAN`, `Midjourney`, `stable_diffusion_v_1_5`, `wukong`). It also showed a
bug: the SD patterns did not accept `stable_diffusion_v_1_5` (underscore between `v` and `1`), and
generators were matched on the whole path, file names included, so SD v1.5's real images were
counted as unassigned. Fix: patterns accept `v_1_4` / `v_1_5`; generators are matched on folder
names only.

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 135 passed, 0 failed, 0 skipped (2 new; both fail on the previous code) |

New tests: the mirror's folder names give 8 generators and 0 unassigned images, and a file whose
name mentions another generator is counted by its folder; with the 5 generators present on Drive
the error names exactly GLIDE, SD v1.4 and VQDM as missing.

## Milestone 2, step 3 (SID-Set verified in place, option A) — 2026-11-12 — branch `claude/nice-curie-3tzxzr`

SID-Set is already on Drive (10,000 per class train, 2,000 per class validation, written by the
retired downloader on 7 Nov 2026). Decision of 12 Nov 2026: verify it in place instead of
downloading it again. New step `verify_sid_folder`; `stage("sidset")` copies the listed files to the
local disk (`storage.copy_files`, resumable, size-checked, space-checked). The probe now also reads
mask values and image/mask pairing for loose mask folders. The parquet download is kept as a tested
fallback (`hf_parquet_subset`).

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 133 passed, 0 failed, 0 skipped (125 before + 8: SID-Set verification failures (6), parquet fallback with staging, `copy_files`) |
| `repo_audit --history` | ok: 54 tracked files, 0 findings |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |

New SID-Set tests: verified in place with no download and no repacking; each of these stops it
with no `_DONE`: a listed file missing, a tampered image without a mask, a mask on a real image,
a class below its cap, a label that disagrees with the folder (label map), a row pointing into
the other split's folder (review of PR #2: no validation image may enter training). `_DONE` now
records the step that produced the data, and `stage()` follows it, so data from the parquet
fallback is unpacked from its tar files (review of PR #2).

## Milestone 2, step 2 (data download: `ensure_data`) — 2026-11-12 — branch `claude/nice-curie-3tzxzr`

Code: `src/eviground/data/{storage,registry,sources,fake_sources,download}.py`, archive support in
`probe.py`, `notebooks/00c_data_download.ipynb`. Follows the storage rules of spec revision 45.
Network code (`sources.NetworkSources`) is not run in the sandbox; every test uses
`fake_sources.FakeSources`, which serves small synthetic zip, tar and parquet files through the
same interface.

### How to run

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
python -m eviground.utils.repo_audit --history
EVIGROUND_DRY_RUN=1 pytest -q --nbmake notebooks/00_setup_and_data.ipynb notebooks/00c_data_download.ipynb notebooks/00b_data_probe.ipynb
```

### Results (sandbox: Python 3.13.18, torch 2.15.0, pyarrow 26.0.1, huggingface_hub 2.4.0, CPU)

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 125 passed, 0 failed, 0 skipped (91 before + 12 storage + 21 download + 1 probe) |
| `repo_audit --history` | ok: 54 tracked files, 0 findings, 16 commits / 69 blobs scanned |
| Notebook dry runs (`00`, `00c`, then `00b`) | 3 passed |
| Downloads on Drive (Colab, CPU runtime) | pending: run `00c_data_download.ipynb` twice, then `00b_data_probe.ipynb` |

### What the new tests cover

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_storage.py` | 12 | Space check message names the step, the disk and how much to free; zip and tar members read without extraction (directory entries ignored); tar shards are byte-reproducible, verified (names and sizes) before they appear, and leave nothing behind on a mismatch; unpacking resumes (skips complete files, redoes partial ones), checks space first and rejects `../` member names |
| `tests/test_download.py` | 21 | Full run on fake sources: GenImage Validation verified in place (8 generators, stated total; the old marker text kept); SD v1.4 subset = seeded 5 + 5 *training* images (never `val`) in 3 tar files with SHA-256 manifest, local archive deleted; DiffSeg30k zips byte-identical to the source, no loose files on Drive; SID-Set caps per class, masks only on tampered, no parquet on Drive. A second run makes no network call. Same seed → byte-identical subsets. Interrupted transfers resume from the partial file (each file transferred exactly twice) and no `_DONE` appears before verification. Shard-level resume: all shards present → no Kaggle download; one shard deleted → only it is rewritten. Too little space on Drive or on the local disk → stops before writing anything. Checksum mismatch → file removed, no `_DONE`. Incomplete Validation → no `_DONE`, original marker untouched. BR-Gen and Chameleon are never downloaded. Unpacking per session and its resume. Kaggle credentials: token → Bearer, legacy key → Basic, none → message naming both options, and no `_DONE` |
| `tests/test_probe.py` | 19 (was 18) | Archive contents listed per folder; DiffSeg30k mask format, mode, distinct values and image/mask pairing read from inside the zips; image statistics for archive members; `not verified` until `ensure_data` has checked a zip; card splits without a zip are `missing`; end-to-end: probe after `ensure_data` reports every item `present` |

## Milestone 2, step 1 (read-only data probe for CP1) — 2026-11-11 — branch `claude/nice-curie-3tzxzr`

Code: `src/eviground/data/imageinfo.py`, `src/eviground/data/probe.py`,
`make_fake_drive()` in `src/eviground/data/synthetic.py`, `notebooks/00b_data_probe.ipynb`.
The probe only reads `Paper 1/datasets/`; the Colab run writes `runs/checkpoints/cp1_probe*.json`.

### How to run

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
python -m eviground.utils.repo_audit --history
EVIGROUND_DRY_RUN=1 pytest -q --nbmake notebooks/00_setup_and_data.ipynb notebooks/00b_data_probe.ipynb
```

### Results (sandbox: Python 3.13.18, torch 2.15.0 from PyPI, CPU)

| Check | Result |
|-------|--------|
| `ruff check src tests` | All checks passed |
| `pytest` | 91 passed, 0 failed, 0 skipped (52 from Milestone 1 + 21 + 18 new) |
| `repo_audit --history` | ok: 45 tracked files, 0 findings, 11 commits / 52 blobs scanned |
| Notebook dry runs (`nbmake`, both notebooks) | 2 passed |
| Probe on Drive (Colab, CPU runtime) | pending: run `notebooks/00b_data_probe.ipynb` |

### What the 39 new tests cover

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_imageinfo.py` | 21 | Format from the file bytes, not the extension (PNG, JPEG, WebP, BMP, TIFF, unknown); JPEG quality estimated exactly for libjpeg files at 30/75/90/95/96/100 and flagged as not exact for custom tables; chroma subsampling 4:4:4/4:2:2/4:2:0; unreadable and missing files reported, not raised; 8×8 blockiness about 1.0 for PNG and clearly higher for JPEG q20; blockiness follows the grid phase of a shifted crop |
| `tests/test_probe.py` | 18 | On a miniature, partially downloaded Drive: folder inventory; GenImage generators found and missing (3 of 8, no SD v1.4 train → `partial` / `missing`); per-folder format, JPEG quality, subsampling, size (<256 count) and blockiness; DiffSeg30k mask format, mode, distinct values, unedited masks, image/mask pairing; SID-Set counts against the 10k/2k caps; Chameleon folder counts equal the zip's; BR-Gen archive parts; the probe changes no file and is deterministic; no network call unless cards are requested; Hugging Face token sent only after a 401 and never stored; network failure recorded, not raised; card split sizes feed the present/missing assessment; after review of PR #1: a split named on the card but absent on Drive is reported `missing`, a generator folder without images does not count as found, a DiffSeg30k split with unpaired image/mask names is not `present`, and a GenImage subset without its `_DONE` marker is `partial` |

## Milestone 1 (environment) — 2026-11-11 — commit `45a9329`

### How to run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CI; Colab already has torch
pip install -e ".[dev]"
ruff check src tests
pytest -q
python -m eviground.utils.repo_audit --history
EVIGROUND_DRY_RUN=1 pytest -q --nbmake notebooks/00_setup_and_data.ipynb
```

### Results

| Check | Where | Result |
|-------|-------|--------|
| `ruff check src tests` | sandbox, Python 3.13.18 | All checks passed |
| `pytest -q` | sandbox, Python 3.13.18, torch 2.15.0 (CPU) | 52 passed, 0 failed, 0 skipped, 2.86 s |
| `repo_audit --history` | sandbox | ok: 38 tracked files, 5 commits and 36 blobs scanned, 0 findings |
| Notebook dry run (`nbmake`) | sandbox | 1 passed |
| GitHub Actions `ci`, Python 3.10 | run 42877556236 | success: lint, tests, audit, notebook dry run |
| GitHub Actions `ci`, Python 3.12 | run 42877556236 | success: lint, tests, audit, notebook dry run |
| Fresh Colab runtime (`notebooks/00_setup_and_data.ipynb`) | Colab CPU runtime, Python 3.13.18, torch 2.14.1, commit `dc6fdc2`, 2026-11-11 | PASS: 52 tests passed, audit ok, Drive mounted (record: Drive `Paper 1/runs/checkpoints/cp0_colab.json`) |

CI run: https://github.com/DHANUSHKUMMARI/Paper1/actions/runs/42877556236

### What the 52 tests cover

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_env.py` | 11 | Paths from environment variables; never a real Drive outside Colab; dry-run override; missing secret names the secret; secrets removed from the environment after use, also on error; `session_info` complete and free of secret values |
| `tests/test_seed.py` | 15 | Seeds 42, 123, 2026 reproduce Python, NumPy and PyTorch numbers and layer initialisation; invalid seeds rejected; deterministic switches set; seeded generator |
| `tests/test_config.py` | 6 | `base.yaml`; `debug.yaml` composes and overrides `base`; overrides; no machine paths in configs; missing and circular configs; save and reload |
| `tests/test_synthetic_fixture.py` | 5 | 12 images per class at 128×128; masks only on edited images; edit lies exactly inside its mask; no duplicate images; same seed gives identical bytes |
| `tests/test_repo_hygiene.py` | 15 | No weights, archives, dataset images or oversized files tracked; no secrets in tracked files or in git history; `.gitignore` blocks data, checkpoints and secrets; required layout exists; the scanner detects planted tokens (also inside `.docx`) and flags a deliberately dirty repository |

### Limits of this sandbox (stated so nothing is assumed)

- No GPU and no access to Kaggle, Hugging Face or Google Drive files. Anything that touches the
  real datasets (CP1 onwards) or a GPU (CP2's GPU validation run, CP3 onwards) runs in Colab.
- `download.pytorch.org` is not reachable from the sandbox, so the local run used the default
  PyPI build of torch; CI uses the CPU build.
