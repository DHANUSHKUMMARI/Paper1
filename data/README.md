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
8. ~~DiffSeg30k's 5,000 training images with an empty mask (unedited) are real negatives.~~
   Revised in the spec review of 14 Nov 2026 (decision 13).
9. ~~Sampler: real = GenImage 1/3, SID-Set 1/3, COCO 1/6, DiffSeg30k-real 1/6.~~ Revised in the
   spec review of 14 Nov 2026 (decision 14); generated = GenImage 2/3, SID-Set 1/3 and
   edited = DiffSeg30k only are unchanged.
10. GenImage's blockiness difference is investigated with two diagnostics: separation on the raw
    files, and the same ratio on the mid-block lines (JPEG has no edges there; an upsampling
    decoder can).

After CP1 run 2 (criteria met, blockiness stop; 14 Nov 2026), Dhanush decided:

11. GenImage's generated-vs-real blockiness (0.6675) is accepted as a generator trace: it is present
    in the raw SD v1.4 PNGs and on mid-block lines, where JPEG puts no edges. It stays reported.
12. One seeded third of DiffSeg30k edited training images is compressed at quality 75, 4:2:0 (like
    SID-Set), so quality-75 images make up one third of every class.

Spec review of 14 Nov 2026 (Dhanush; spec cards 2, 5, 6 and 8 on hold until CP1 run 4):

13. DiffSeg30k's unedited training images (empty mask) leave training. About half of DiffSeg30k's
    base images are AI-generated (paper section 3.1), and nothing in the release says which:
    `metadata.jsonl` holds only `image_file_name` and `mask_file_name` (first lines of both
    files, CP1 run-3 record); the zips hold no other metadata. CP1 run 4 counts the keys on every
    line and stops and asks if any other key appears. Image size is reported as an indication
    only (1024×1024 vs both sides ≤ 640). They are not relabelled as fully generated.
    Run 4 (14 Nov 2026): every line of both files (28,998 + 6,001) holds only the two file names,
    0 unparsable. All 5,000 unedited images have both sides ≤ 640 (COCO size); Dhanush kept them
    out, since size is not a field.
14. Compression does not depend on the class inside a source: the same seeded third (seed 42) of
    every DiffSeg30k and COCO training image, whatever its class, is compressed at quality 75,
    4:2:0; the rest keep the GenImage real distribution. Sampler: real = GenImage 1/3,
    SID-Set 1/6, COCO 1/2; generated = GenImage 2/3, SID-Set 1/3; edited = DiffSeg30k only.
    Quality 75 is then one third of every class (real 1/6 + 1/2 × 1/3), plus the about 5% of
    quality-75 files in the GenImage real distribution itself, which every class gets alike.

Staged copies: `stage()` records every local file with its size (`_STAGED_FILES.csv.gz`); CP1
starts only if the local disk matches that list exactly (none missing, none with another size,
no extra files).

After notebook 03 (mask labels, 14 Nov 2026), Dhanush decided (option 1, strict hold-out):

15. DiffSeg30k labels an AI-generated base with its generator's value on every pixel (15,687 of
    34,999 masks have no 0). That base value is read as 0 (`diffseg_mask.read_base`): it is the
    value covering most of the image border and must also be the most frequent value; if the two
    differ the image leaves training (counted; CP1 stops and asks above 5% of full masks). A mask
    with no edit left once the base is removed (606) is dropped, never relabelled.
16. Glide (6) and Kolors (2) stay out of training wherever they appear, base or edit. P2 = the
    validation images where Glide or Kolors made an edit; P2 is reported on all images and on the
    real-base subset (only the subset is compared with the DiffSeg30k paper: there both mask
    readings agree). Manifest columns `mask_base`, `base_type`, `base_reading`, `edit_values`.
17. DiffSeg30k's 5,000 unedited images (all-zero mask) are real negatives again: with this
    encoding an AI-generated base never has an all-zero mask, so they are real COCO photos (all
    COCO-sized, CP1 run 4). Real = GenImage 1/3, SID-Set 1/6, COCO 1/4, DiffSeg30k 1/4; the same
    seeded third of every DiffSeg30k and COCO image stays at quality 75, so quality 75 is one
    third of every class. Decision 13 is reversed.

CP1 run 5 samples edited crops with the training rule (`crops.edited_crop`) on the read masks and
replaces failed draws; the failed-draw histogram is reported before the 1% minimum is changed.

## Harmonization of the training data (spec section 8, M1, revision 98)

Within each source every class ends with the compression history of its most compressed class.
Compression happens once, on the whole image, before the 256 crop
(`eviground.data.harmonize.plan_image`; the same image always gets the same quality, seed 42).

| Source | Harmonization |
|---|---|
| GenImage SD v1.4 | Generated PNGs -> JPEG with (quality, subsampling) drawn from the measured distribution of the real training JPEGs; real images untouched |
| SID-Set | Real, synthetic and PNG tampered -> JPEG quality 75, 4:2:0; images already at 75 / 4:2:0 untouched; JPEGs below 75 skipped and counted |
| DiffSeg30k and COCO negatives | All PNG after preparation -> JPEG with the GenImage real distribution, except one seeded third of each source (any class) at quality 75, 4:2:0 (decision 14) |

Images with a side under 256 are skipped and counted. Sampler: see decision 14 above (spec rev 49
had real = GenImage, SID-Set, COCO one third each; edited = DiffSeg30k two thirds, SID-Set one third). DiffSeg30k training images contain only the 6 training
editors (mask values 1, 3, 4, 5, 7, 8); validation images with any Glide (6) or Kolors (2) pixel
are the P2 test set. Validation: 10% of each training source, stratified, seed 42.

Leftovers of the retired downloader (`notebooks/tools/README.md`) are left untouched and are
reported by the probe as "legacy": the empty DiffSeg30k folders and header-only manifests.

## Training crops (decision of 14 Nov 2026; minimum pending)

- Edited images (`crops.edited_crop`): pick a random edited pixel, then a 256 window drawn
  uniformly among all in-image windows that contain it (no centring: test tiles have no centre
  prior); keep it if the edited share is 1–90%; up to 10 tries. A failed image is replaced by
  another from the same (class, source) pool (`epoch.epoch_plan`), so 1 : 1 : 1 holds; failures
  and replacements are logged; images with no valid window at all (`epoch.screen_never_valid`)
  are left out and logged.
- CP1 keeps its own rule (`crops.edited_box`, a uniform window) so runs 1–4 stay reproducible.
- The 1% minimum stays. `notebooks/02_crop_diagnostic.ipynb` (14 Nov 2026, run 4 reproduced:
  1,585 of 3,000 edited draws failed) found that the failures are not small edits: 1,583 had
  every try above 90% and only 2 below 1%; a 256-pixel minimum would change 1,581 failures to
  1,580. 5,930 of the 11,382 edited training images have no valid window; their masks have no
  zero pixel (median edited share 100%, 1,048,576 pixels = 1024 × 1024). Spec section 8 assumes
  the background of AI-generated bases is labelled 0. `notebooks/03_mask_diagnostic.ipynb`
  measures what the mask values mean before anything is decided.
- Validation accuracy is logged per source (`eviground.eval.metrics`), to see whether COCO, half
  of the real class, is memorised.

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
4. `notebooks/02_crop_diagnostic.ipynb`: copies DiffSeg30k to the local disk and rebuilds CP1
   run 4's edited crop draws from its manifest; writes `runs/checkpoints/crop_diagnostic.json`.
5. `notebooks/03_mask_diagnostic.ipynb`: copies DiffSeg30k to the local disk and measures every
   mask's values (read-only); writes `runs/checkpoints/mask_diagnostic.json` and
   `mask_examples.png`.

Cite each dataset's original paper.
