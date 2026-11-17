# Testing log

Commands and results for every milestone, newest first. Everything here runs on CPU with
synthetic data: no GPU, no Kaggle or Hugging Face access, no real images.

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
