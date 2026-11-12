# Testing log

Commands and results for every milestone, newest first. Everything here runs on CPU with
synthetic data: no GPU, no Kaggle or Hugging Face access, no real images.

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
