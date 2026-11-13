"""Dataset registry: where each dataset comes from, where it lives on Drive, how it is fetched.

Folders are relative to ``Paper 1/datasets/``. Storage rules (Research Specification,
section 13, revision 45): source archives stay on Drive and are unpacked on the local disk
each session; no loose-file copies and no duplicate shards on Drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LABELS = {"real": 0, "generated": 1, "edited": 2}


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset. ``method`` selects the step in :mod:`eviground.data.download`."""

    key: str
    folder: str
    method: str  # verify_folder | verify_sid_folder | kaggle_subset | hf_files | url_files |
    #              hf_parquet_subset | deferred
    source: str
    role: str
    licence: str
    citation: str
    params: dict[str, Any] = field(default_factory=dict)


REGISTRY: dict[str, DatasetSpec] = {
    "genimage_val": DatasetSpec(
        key="genimage_val",
        folder="01_GenImage/Validation",
        method="verify_folder",
        source="kaggle:vtphatt2/GenImage-Validation (5 generators already on Drive as loose files)",
        role="P1 test set: 8 generators, real (nature) and generated (ai)",
        licence="Non-commercial research",
        citation="Zhu et al., GenImage, NeurIPS 2023 Datasets and Benchmarks",
        params={
            "expected_images": 100_000,
            "generators": 8,
            # Drive held 5 of 8 generators (12 Nov 2026); the missing ones (GLIDE, SD v1.4,
            # VQDM) are added from this Kaggle archive as loose files in the same layout as
            # the other five (decision of 12 Nov 2026). None = verify only.
            "fill_from": "vtphatt2/GenImage-Validation",
            "fill_gb_estimate": 12.0,
            "class_dirs": {"nature": LABELS["real"], "ai": LABELS["generated"]},
        },
    ),
    "genimage_sd14": DatasetSpec(
        key="genimage_sd14",
        folder="01_GenImage/SD_v1.4_subset",
        method="kaggle_subset",
        source="kaggle:vtphatt2/GenImage-stable-diffusion-v1-4",
        role="Training data of both models: seeded subset of the SD v1.4 training split",
        licence="Non-commercial research",
        citation="Zhu et al., GenImage, NeurIPS 2023 Datasets and Benchmarks",
        params={
            "per_class": 40_000,
            "seed": 42,
            "split_dir": "train",
            "class_dirs": {"nature": LABELS["real"], "ai": LABELS["generated"]},
            "shard_images": 10_000,
            "archive_gb": 89.8,  # stated size, used only for the first space check
            "subset_gb_estimate": 25.0,  # replaced by the exact size once listed
        },
    ),
    "diffseg30k": DatasetSpec(
        key="diffseg30k",
        folder="02_DiffSeg30k",
        method="hf_files",
        source="hf:Chaos2629/Diffseg30k",
        role="Locally edited images + masks; editor hold-out (Glide, Kolors) for P2",
        licence="Apache-2.0",
        citation="DiffSeg30k (arXiv 2511.19111)",
        params={"files": ["train.zip", "validation.zip"]},
    ),
    "sidset": DatasetSpec(
        key="sidset",
        folder="03_SID-Set",
        # Already on Drive (10k/class train, 2k/class validation, written by the retired
        # downloader): verified in place, never downloaded again (decision of 12 Nov 2026).
        # "hf_parquet_subset" is kept as a tested fallback that downloads into tar files.
        method="verify_sid_folder",
        source="hf:saberzl/SID_Set",
        role="Real / fully synthetic / tampered (+ masks); train subset for training, "
        "validation subset as the P3b test set",
        licence="CC-BY-4.0",
        citation="Huang et al., SIDA, CVPR 2025",
        params={
            "caps": {"train": 10_000, "validation": 2_000},
            "seed": 42,
            "labels": {0: "real", 1: "full_synthetic", 2: "tampered"},
            "tampered_label": 2,
            "label_column": "label",
            "image_column": "image",
            "mask_column": "mask",
            "id_column": "img_id",
        },
    ),
    "coco_val2017": DatasetSpec(
        key="coco_val2017",
        folder="06_COCO",
        method="url_files",
        source="https://cocodataset.org (images.cocodataset.org)",
        role="Real negatives for DiffSeg30k (about 4,850, prepared like its base images) and the "
        "150 real photos reserved for the Claims-Edit case study (decision of 13 Nov 2026)",
        licence="COCO images: Flickr terms of use; annotations CC-BY-4.0",
        citation="Lin et al., Microsoft COCO, ECCV 2014",
        params={
            # COCO publishes no checksums: each zip is CRC-tested, its members counted and its
            # SHA-256 recorded on the first verification.
            "files": {
                "val2017.zip": {
                    "url": "http://images.cocodataset.org/zips/val2017.zip",
                    "members": {"prefix": "val2017/", "suffix": ".jpg", "count": 5000},
                    "gb_estimate": 0.82,
                },
                "annotations_trainval2017.zip": {
                    "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                    "required": ["annotations/instances_val2017.json"],
                    "gb_estimate": 0.26,
                },
            },
            "stage": ["val2017.zip"],  # the annotations are read from the zip, not unpacked
            "claims_edit": {
                "n": 150,
                "categories": ["car", "couch", "chair", "dining table", "laptop"],
                "min_area_share": 0.05,
                "seed": 42,
            },
            "negatives_size": 1024,
        },
    ),
    "brgen": DatasetSpec(
        key="brgen",
        folder="04_BR-Gen",
        method="deferred",
        source="hf:clpbc/BR-Gen (68.5 GB split zip)",
        role="P3 cross-dataset test only; not needed for CP1. A subset plan is proposed before P3",
        licence="See dataset card",
        citation="BR-Gen / NFA-ViT (arXiv 2504.11922)",
    ),
    "chameleon": DatasetSpec(
        key="chameleon",
        folder="05_Chameleon",
        method="deferred",
        source="manual (AIDE repository links)",
        role="P4 hard test set; counts checked by the data probe",
        licence="See source",
        citation="Yan et al., A Sanity Check for AI-generated Image Detection, ICLR 2025",
    ),
}

# What ensure_data() fetches or verifies per profile (CP1 needs these four).
PROFILES: dict[str, list[str]] = {
    "medium": ["genimage_val", "genimage_sd14", "diffseg30k", "sidset"],
    # CP1 adds the COCO real negatives (decision of 13 Nov 2026)
    "cp1": ["genimage_val", "genimage_sd14", "diffseg30k", "sidset", "coco_val2017"],
}
