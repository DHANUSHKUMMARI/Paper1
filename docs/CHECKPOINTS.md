# Verification checkpoints (CP0–CP10)

Criteria are defined in section 9.1 of the Research Specification. The day for each checkpoint is in the 15-day execution plan.

Claude Code: after each checkpoint, fill in its entry below. Record what was run, the exact numbers, the run folder or commit, and PASS or FAIL against the criteria. Do not start the next stage while a checkpoint is FAIL. Dhanush verifies each entry and initials it.

| CP | Day | Stage | Status | Verified by |
|----|-----|-------|--------|-------------|
| CP0 | 1 | Environment | **PASS** (11 Nov 2026) | |
| CP1 | 2 | Data integrity and leakage | pending | |
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
- **Verified by Dhanush:** ☐
