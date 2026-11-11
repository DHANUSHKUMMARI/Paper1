# Testing log

Commands and results for every milestone, newest first. Everything here runs on CPU with
synthetic data: no GPU, no Kaggle or Hugging Face access, no real images.

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
