# Small manifests (committed)

Copied verbatim from the CP1 run-3 record `Paper 1/runs/checkpoints/cp1_data_integrity.json`
(written 2026-11-14T11:19:39+00:00, `main` at `git 2feec5a`, verdict PASS). The full image manifest of that run
(`manifest_cp1.csv.gz`, every image with hashes, roles, splits and harmonization plan) stays on
Drive in `Paper 1/runs/checkpoints/cp1/`; it is too large for the repository.

- `claims_edit_reserved_coco_val2017.csv`: the 150 COCO val2017 photos reserved for the
  Claims-Edit case study (P7) and never used for training. Rule: a non-crowd instance of
  car, couch, chair, dining table, laptop covering at least 5% of the image; the categories take turns
  drawing from a seeded permutation (seed 42) (`eviground.data.cp1.reserve_claims_edit`).
- `genimage_sd14_real_quality.json`: the measured (JPEG quality, chroma subsampling) distribution of
  the GenImage SD v1.4 real training images (35,302 colour JPEGs; quality 96 in
  66.5%). GenImage generated, DiffSeg30k and COCO images are compressed with
  draws from it (`eviground.data.harmonize.QualityDistribution`).
