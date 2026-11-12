# Tools

`Paper1_Dataset_Downloader_v3_medium.ipynb` is **retired** (12 Nov 2026). Do not run it. Known faults:

- DiffSeg30k: it assumed a parquet dataset; the repository holds `train.zip` (21.6 GB) and
  `validation.zip` (4.88 GB), so only empty manifest files were written.
- GenImage SD v1.4: it unpacked the whole 89.8 GB archive on Colab's disk and ran out of space.
- BR-Gen: it copied the 68.5 GB split archive to Drive without checking that it would fit.

Downloads are now handled by `eviground.data.download.ensure_data()` (Milestone 2), following the
storage rules in section 13 of `docs/RESEARCH_SPEC.md`.
