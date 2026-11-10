# Paper1 — EviGround

**EviGround: Evidence-Grounded, Hallucination-Aware Explainable Detection and Localization of AI-Generated and AI-Edited Images**

Research code for Paper 1. The model classifies an image as *real*, *fully AI-generated* or *locally AI-edited*, localizes edited regions, and produces a forensic report written by a frozen LLM that may cite **only** ID-tagged detector evidence, checked by a rule-based verifier.

> Status: Milestone 1 (environment) built; checkpoint CP0 passes in CI and awaits the Colab run.
> Implementation follows `docs/IMPLEMENTATION_BRIEF.docx` (milestones 1–8); progress is recorded in
> `docs/CHECKPOINTS.md`, `docs/TESTING.md` and `docs/EXPERIMENT_LOG.md`.

Target journals (hybrid, subscription route, no APC), in order: Multimedia Systems; Pattern Analysis
and Applications; The Visual Computer; Applied Intelligence; International Journal of Multimedia
Information Retrieval (all Springer).

## Where things live

| What | Where |
|---|---|
| Code, configs, notebooks, tests | this repository |
| Datasets (~60–100 GB) | Google Drive: `MyDrive/Paper 1/datasets/` — **not in Git** (see `data/README.md`) |
| Packed data shards | Google Drive: `MyDrive/Paper 1/cache/shards/` |
| Checkpoints, logs, results | Google Drive: `MyDrive/Paper 1/runs/` and `outputs/` |
| Secrets (HF, Kaggle, GitHub tokens) | Colab **Secrets** panel only — never in files |

## Running on Google Colab Pro

1. Open a notebook from `notebooks/` in Colab.
2. Add secrets in the Colab Secrets panel: `GITHUB_TOKEN` (read-only), and for the first data download only `HF_TOKEN` and `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`).
3. Run Cell 1 (mount Drive, clone/pull this repo, install, check data). Data is downloaded from Kaggle / Hugging Face **only if it is missing from Drive**.
4. Training checkpoints to Drive and resumes automatically after a disconnect.

## Tests

```bash
pip install -e ".[dev]"
ruff check src tests && pytest -q
python -m eviground.utils.repo_audit --history   # no secrets, datasets or checkpoints in Git
```

## Repository layout

```
configs/      YAML configs (data, model, explain, experiments, debug)
data/         README + small manifest files (no images)
notebooks/    Colab entry points; tools/ has the dataset downloader
src/eviground/  package: data, models (M1–M4), explain (M5–M7), train, eval, baselines
scripts/      command-line entry points
tests/        CPU tests on synthetic data (run in CI)
docs/         implementation brief, testing and experiment logs
```

## Citation

To be added at submission.
