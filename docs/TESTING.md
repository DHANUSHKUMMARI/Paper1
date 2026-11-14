# Testing log

Commands and results for every milestone, newest first. Everything here runs on CPU with
synthetic data: no GPU, no Kaggle or Hugging Face access, no real images.

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
