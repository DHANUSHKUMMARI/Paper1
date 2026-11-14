# Datasets

The datasets are **not stored in this repository**:

- **Size:** tens of GB. GitHub rejects files over 100 MB, and Git LFS storage would exceed the project budget.
- **Licences:** GenImage is released for non-commercial research and SID-Set (CC-BY-4.0) requires attribution; redistributing copies here could breach their terms.

Instead, the data is rebuilt on Google Drive from the original sources by
`eviground.data.download.ensure_data()` (notebook `notebooks/00c_data_download.ipynb`). This
folder keeps only small files needed to reproduce it exactly (`manifests/`).

## Storage rules (Research Specification, section 13, revision 45)

- Free space on **both** Drive and Colab's local disk (about 115 GB) is checked before every
  download and before every shard; a step that would not fit stops with a message and writes
  nothing. On Colab the Drive mount reports the machine's own disk as its free space, so the Drive
  side is read from the Google Drive quota through the Drive API. Before each write to Drive the
  step also waits for the local Drive upload cache to have room.
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
| `chameleon` | Manual (AIDE repository links) | `05_Chameleon/` | Partly unpacked by hand (1,412 of 26,033 images, 13 Nov 2026); the zip is being put back. Not needed for CP1 | See source | P4 hard test set |
| `coco_val2017` | images.cocodataset.org (`val2017.zip`, `annotations_trainval2017.zip`) | `06_COCO/` | Both zips as published, CRC-tested, 5,000 images counted, SHA-256 recorded; only the images are unpacked locally | Images: Flickr terms; annotations CC-BY-4.0 | 150 photos reserved for Claims-Edit (P7); the other ~4,850 are real negatives for DiffSeg30k, each at a size drawn from the measured sizes of DiffSeg30k's training images (Lanczos resize by the smallest factor that covers it, centre crop; decision of 14 Nov 2026) |

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

- **COCO Claims-Edit reservation** (before anything else): images with a non-crowd instance of
  car, couch, chair, dining table or laptop covering at least 5% of the image; the categories take
  turns drawing from a seeded permutation (seed 42) until 150 are reserved. They are a test set
  (P7) and never used for training. COCO negatives with a near-duplicate (pHash Hamming <= 6, raw or
  prepared) in any test set are dropped.

## CP1 decisions confirmed by Dhanush (14 Nov 2026)

1. COCO negatives match DiffSeg30k's measured image sizes (not a fixed 1024 × 1024 crop).
2. Claims-Edit reservation: 150 COCO photos with a non-crowd car, couch, chair, dining table or
   laptop covering at least 5% of the image; categories take turns, seed 42.
3. The 60% metadata criterion uses 5-fold cross-validated balanced accuracy (real vs not real).
4. Near-duplicate: pHash Hamming distance at most 6 (counts at 0–10 are reported).
5. Blockiness "clear class difference": a single threshold separates two classes with
   cross-validated balanced accuracy above 0.60.

After the first CP1 run (FAIL, 14 Nov 2026; see `docs/CHECKPOINTS.md`), Dhanush decided:

6. Training images whose file or pixel hash equals a test image's, or with a near-duplicate
   (pHash ≤ 6) in any test set, are left out of training; test sets stay as published. Identical
   images inside the training pool go to the same split.
7. SID-Set tampered images are left out of training (the agreed blockiness fallback); SID-Set
   validation (P3b) keeps them.
8. DiffSeg30k's 5,000 training images with an empty mask (unedited) are real negatives; COCO
   negatives are kept.
9. Sampler: real = GenImage 1/3, SID-Set 1/3, COCO 1/6, DiffSeg30k-real 1/6; generated = GenImage
   2/3, SID-Set 1/3; edited = DiffSeg30k only.
10. GenImage's blockiness difference is investigated with two diagnostics: separation on the raw
    files, and the same ratio on the mid-block lines (JPEG has no edges there; an upsampling
    decoder can).

Staged copies: `stage()` records every local file with its size (`_STAGED_FILES.csv.gz`); CP1
starts only if the local disk matches that list exactly (none missing, none with another size,
no extra files).

## Harmonization of the training data (spec section 8, M1, revision 49)

Within each source every class ends with the compression history of its most compressed class.
Compression happens once, on the whole image, before the 256 crop
(`eviground.data.harmonize.plan_image`; the same image always gets the same quality, seed 42).

| Source | Harmonization |
|---|---|
| GenImage SD v1.4 | Generated PNGs -> JPEG with (quality, subsampling) drawn from the measured distribution of the real training JPEGs; real images untouched |
| SID-Set | Real, synthetic and PNG tampered -> JPEG quality 75, 4:2:0; images already at 75 / 4:2:0 untouched; JPEGs below 75 skipped and counted |
| DiffSeg30k and COCO negatives | All PNG after preparation -> JPEG with the GenImage real distribution |

Images with a side under 256 are skipped and counted. Sampler: see decision 9 above (spec rev 49
had real = GenImage, SID-Set, COCO one third each; edited = DiffSeg30k two thirds, SID-Set one third). DiffSeg30k training images contain only the 6 training
editors (mask values 1, 3, 4, 5, 7, 8); validation images with any Glide (6) or Kolors (2) pixel
are the P2 test set. Validation: 10% of each training source, stratified, seed 42.

Leftovers of the retired downloader (`notebooks/tools/README.md`) are left untouched and are
reported by the probe as "legacy": the empty DiffSeg30k folders and header-only manifests.

## Notebooks

1. `notebooks/00c_data_download.ipynb`: run twice, each time in a fresh CPU runtime: first
   `['genimage_val', 'diffseg30k', 'sidset']` (only DiffSeg30k is downloaded), then
   `['genimage_sd14']`. The SD v1.4 archive
   needs about 90 GB of the local disk.
2. `notebooks/00b_data_probe.ipynb`: read-only; records what is on Drive, including the
   contents of the zip and tar files, mask formats and values, JPEG qualities and blockiness.
3. `notebooks/01_cp1_data_integrity.ipynb`: downloads COCO once, stages everything locally and
   runs checkpoint CP1 (`eviground.data.cp1.run_cp1`); writes
   `runs/checkpoints/cp1_data_integrity.json`, the sample grids and `manifest_cp1.csv.gz`.

Cite each dataset's original paper.
