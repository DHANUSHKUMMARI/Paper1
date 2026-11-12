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
| `genimage_val` | Kaggle `vtphatt2/GenImage-Validation` (already on Drive) | `01_GenImage/Validation/` | The existing folder (100,000 images, 25.5 GB), verified in place, never re-downloaded | Non-commercial research | P1 test set (8 generators) |
| `genimage_sd14` | Kaggle `vtphatt2/GenImage-stable-diffusion-v1-4` (89.8 GB archive) | `01_GenImage/SD_v1.4_subset/` | `sd14_subset-0000k.tar` (a few files), `subset_plan.csv`, `manifest.csv` (with SHA-256). The archive is downloaded to the local disk without extraction and deleted afterwards | Non-commercial research | Training: seeded subset of 40,000 real + 40,000 generated training images (seed 42) |
| `diffseg30k` | Hugging Face `Chaos2629/Diffseg30k` | `02_DiffSeg30k/` | `train.zip` (21.6 GB) and `validation.zip` (4.88 GB) as published, SHA-256 checked | Apache-2.0 | Local AI edits + masks; Glide/Kolors editor hold-out (P2) |
| `sidset` | Hugging Face `saberzl/SID_Set` (parquet) | `03_SID-Set/` | `sid_<split>-<source>.tar` files, `<split>_manifest.csv`, `<split>_plan.json`: 10,000 per class from train, 2,000 per class from validation | CC-BY-4.0 | Real / synthetic / tampered; validation = P3b test set |
| `brgen` | Hugging Face `clpbc/BR-Gen` (68.5 GB split zip) | `04_BR-Gen/` | Not downloaded for CP1; a subset plan is proposed before P3 | See dataset card | P3 cross-dataset test |
| `chameleon` | Manual (AIDE repository links) | `05_Chameleon/` | Unpacked by hand (`test/0_real`, `test/1_fake`); counts checked by the probe | See source | P4 hard test set |

How the subsets are chosen:

- **GenImage SD v1.4:** members of the archive under `train/nature` (real) and `train/ai`
  (generated) are listed. 40,000 of each are drawn with `numpy.random.default_rng(42)` from the
  name-sorted list, then written in archive order into tar files of 10,000 images.
- **SID-Set:** the `label` column of every parquet file of a split is read remotely (only that
  column). Files are then taken whole in a seeded order until each class reaches its cap; only the
  last file used for a class is subsampled, so few files are downloaded.

Leftovers of the retired downloader (`notebooks/tools/README.md`) are left untouched and are
reported by the probe as "legacy": empty DiffSeg30k folders and header-only manifests, and the
loose SID-Set files.

## Notebooks

1. `notebooks/00c_data_download.ipynb`: run twice, each time in a fresh CPU runtime: first
   `['genimage_val', 'diffseg30k', 'sidset']`, then `['genimage_sd14']`. The SD v1.4 archive
   needs about 90 GB of the local disk.
2. `notebooks/00b_data_probe.ipynb`: read-only; records what is on Drive, including the
   contents of the zip and tar files, mask formats and values, JPEG qualities and blockiness.

Cite each dataset's original paper.
