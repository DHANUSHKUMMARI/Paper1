# Datasets

The datasets are **not stored in this repository**:

- **Size:** tens of GB. GitHub rejects files over 100 MB, and Git LFS storage would exceed the project budget.
- **Licences:** GenImage is released for non-commercial research and SID-Set (CC-BY-4.0) requires attribution; redistributing copies here could breach their terms.

Instead, the data is rebuilt on Google Drive from the original sources by
`eviground.data.download.ensure_data()` (notebook `notebooks/00c_data_download.ipynb`). This
folder keeps only small files needed to reproduce it exactly (`manifests/`).

## Storage rules (Research Specification, section 13, revision 45)

- Drive has about 110 GB free and Colab's local disk about 115 GB. Free space on **both** is checked
  before every download and before every shard; a step that would not fit stops with a message
  and writes nothing.
- Source archives stay on Drive and are unpacked on the local disk each session
  (`eviground.data.download.stage()`). No loose-file copies and no duplicate shards on Drive.
- Every download step is resumable, verifies file counts, sizes (and checksums where the source
  publishes them) afterwards, and writes `_DONE` (JSON, `"verified": true`) only after that.
  A dataset with a valid `_DONE` is skipped with no network call and no secret read.

## Datasets and Drive layout (`MyDrive/Paper 1/datasets/`)

| Key | Source | Drive folder | What is stored | Licence | Role |
|---|---|---|---|---|---|
| `genimage_val` | Kaggle `vtphatt2/GenImage-Validation` | `01_GenImage/Validation/` | 5 generators already on Drive as loose files (`GenImage/<Generator>/val/{ai,nature}`: ADM, BigGAN, Midjourney, SD v1.5, Wukong; 60,411 images). The missing GLIDE, SD v1.4 and VQDM are taken from the Kaggle archive (downloaded to the local disk, not extracted, then deleted) and written as loose files in the same layout (`GenImage/<Generator>/val/{ai,nature}`), with `fill_plan.csv` and `fill_manifest_<generator>.json` (SHA-256 per file). Verified: 8 generators, 100,000 images; filled files are re-checked by size on every run | Non-commercial research | P1 test set (8 generators) |
| `genimage_sd14` | Kaggle `vtphatt2/GenImage-stable-diffusion-v1-4` (89.8 GB archive) | `01_GenImage/SD_v1.4_subset/` | `sd14_subset-0000k.tar` (a few files), `subset_plan.csv`, `manifest.csv` (with SHA-256). The archive is downloaded to the local disk without extraction and deleted afterwards | Non-commercial research | Training: seeded subset of 40,000 real + 40,000 generated training images (seed 42) |
| `diffseg30k` | Hugging Face `Chaos2629/Diffseg30k` | `02_DiffSeg30k/` | `train.zip` (21.6 GB) and `validation.zip` (4.88 GB) as published, SHA-256 checked | Apache-2.0 | Local AI edits + masks; Glide/Kolors editor hold-out (P2) |
| `sidset` | Hugging Face `saberzl/SID_Set` (already on Drive) | `03_SID-Set/` | The existing loose files from the retired downloader, `<split>/<class>/images/` and tampered `masks/`, with `train_manifest.csv` and `validation_manifest.csv`: 10,000 per class from train, 2,000 per class from validation. Verified in place, never downloaded again; copied to the local disk each session | CC-BY-4.0 | Real / synthetic / tampered; validation = P3b test set |
| `brgen` | Hugging Face `clpbc/BR-Gen` (68.5 GB split zip) | `04_BR-Gen/` | Not downloaded for CP1; a subset plan is proposed before P3 | See dataset card | P3 cross-dataset test |
| `chameleon` | Manual (AIDE repository links) | `05_Chameleon/` | Unpacked by hand (`test/0_real`, `test/1_fake`); counts checked by the probe | See source | P4 hard test set |

How the subsets are chosen:

- **GenImage SD v1.4:** members of the archive under `train/nature` (real) and `train/ai`
  (generated) are listed. 40,000 of each are drawn with `numpy.random.default_rng(42)` from the
  name-sorted list, then written in archive order into tar files of 10,000 images.
- **SID-Set** (already on Drive, 7 Nov 2026): the retired downloader streamed each split
  (249 train and 34 validation parquet files) through `datasets` with `shuffle(seed=42,
  buffer_size=300)` and kept the first 10,000 (train) or 2,000 (validation) images of each class,
  with the mask of every tampered image. Its log reported exactly `{0: 10000, 1: 10000, 2: 10000}`
  and `{0: 2000, 1: 2000, 2: 2000}`. `ensure_data()` verifies this in place: per-class counts,
  every listed file present and non-empty, no duplicates, folder and `label_name` agreeing with
  the label map, a mask on every tampered image and none on the others. If a re-download is ever
  needed, `hf_parquet_subset` (kept and tested) takes whole parquet files in a seeded order into
  tar files.

Leftovers of the retired downloader (`notebooks/tools/README.md`) are left untouched and are
reported by the probe as "legacy": the empty DiffSeg30k folders and header-only manifests.

## Notebooks

1. `notebooks/00c_data_download.ipynb`: run twice, each time in a fresh CPU runtime: first
   `['genimage_val', 'diffseg30k', 'sidset']` (only DiffSeg30k is downloaded), then
   `['genimage_sd14']`. The SD v1.4 archive
   needs about 90 GB of the local disk.
2. `notebooks/00b_data_probe.ipynb`: read-only; records what is on Drive, including the
   contents of the zip and tar files, mask formats and values, JPEG qualities and blockiness.

Cite each dataset's original paper.
