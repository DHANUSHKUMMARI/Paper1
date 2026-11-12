"""Tiny synthetic three-class dataset for tests, CI and CPU smoke runs.

No real images and no downloads. Classes follow the project label map:
0 = real, 1 = fully generated, 2 = locally edited (with a ground-truth mask).

- real: smooth gradients plus sensor-like noise;
- generated: the same kind of content with a periodic 8-pixel grid (an upsampling-style trace);
- edited: a real-style image with one square region replaced by generated-style texture.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = {"real": 0, "generated": 1, "edited": 2}
MANIFEST_FIELDS = ("path", "label", "mask", "base", "source")


def _real(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    channels = []
    for _ in range(3):
        a, b, c = rng.uniform(0.2, 1.0, size=3)
        channels.append(120 + 80 * np.sin(2 * np.pi * (a * xx + b * yy) + c))
    img = np.stack(channels, axis=-1) + rng.normal(0, 4.0, size=(size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _generated(rng: np.random.Generator, size: int) -> np.ndarray:
    img = _real(rng, size).astype(np.float32)
    grid = np.zeros((size, size), dtype=np.float32)
    grid[::8, :] += 10.0
    grid[:, ::8] += 10.0
    return np.clip(img + grid[..., None], 0, 255).astype(np.uint8)


def make_synthetic_dataset(
    root: Path | str, per_class: int = 12, size: int = 128, seed: int = 42
) -> list[dict[str, object]]:
    """Write the dataset under ``root`` (lossless PNG) and return its records.

    The same ``seed`` always produces byte-identical files. A ``manifest.csv`` with the
    returned records (relative paths) is written at the top of ``root``.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    for folder in ("real", "generated", "edited", "edited_base", "masks"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for i in range(per_class):
        rel = f"real/{i:04d}.png"
        Image.fromarray(_real(rng, size)).save(root / rel)
        records.append(
            {"path": rel, "label": 0, "mask": "", "base": "", "source": "synthetic/real"}
        )

    for i in range(per_class):
        rel = f"generated/{i:04d}.png"
        Image.fromarray(_generated(rng, size)).save(root / rel)
        records.append(
            {"path": rel, "label": 1, "mask": "", "base": "", "source": "synthetic/generated"}
        )

    side = size // 4
    for i in range(per_class):
        base = _real(rng, size)
        patch = _generated(rng, size)
        y0, x0 = (int(v) for v in rng.integers(0, size - side, size=2))
        edited = base.copy()
        edited[y0 : y0 + side, x0 : x0 + side] = patch[y0 : y0 + side, x0 : x0 + side]
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[y0 : y0 + side, x0 : x0 + side] = 255
        rel, mask_rel, base_rel = (
            f"edited/{i:04d}.png",
            f"masks/{i:04d}.png",
            f"edited_base/{i:04d}.png",
        )
        Image.fromarray(edited).save(root / rel)
        Image.fromarray(mask).save(root / mask_rel)
        Image.fromarray(base).save(root / base_rel)
        records.append(
            {
                "path": rel,
                "label": 2,
                "mask": mask_rel,
                "base": base_rel,
                "source": "synthetic/edited",
            }  # fmt: skip
        )

    with open(root / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return records


# --------------------------------------------------------------------------------------
# Miniature "Paper 1/datasets" folder in the layout of the v3 downloader (medium profile).
# Used by the probe tests and the notebook dry runs; deliberately incomplete, like a
# partially downloaded Drive: GenImage has 3 of 8 generators and no SD v1.4 train set.
# --------------------------------------------------------------------------------------

FAKE_GENERATORS = ("imagenet_ai_0508_adm", "imagenet_glide", "imagenet_ai_0419_biggan")
FAKE_MASK_VALUES = {"train": (1, 3, 5), "validation": (2, 6, 8)}


def _save(arr: np.ndarray, path: Path, fmt: str, **kw: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, fmt, **kw)


def make_fake_drive(datasets_root: Path | str, per_folder: int = 4, seed: int = 42) -> Path:
    """Write a tiny partial copy of the Drive datasets folder; return its path.

    - 01_GenImage/Validation/<generator>/val/{ai: PNG 320x320, nature: JPEG q96 4:2:0,
      one of them 200 px on its short side}; ``_DONE`` holds the Kaggle slug, as the
      downloader writes it. No SD_v1.4.
    - 02_DiffSeg30k/{train,validation}.zip with <split>/{images,masks}/ inside (mask values
      per split from ``FAKE_MASK_VALUES``, one all-zero mask), plus the empty folders and
      header-only manifests the retired downloader left.
    - 03_SID-Set/<split>/<label>/images (+ masks for tampered) and manifests.
    - 04_BR-Gen: two archive parts. 05_Chameleon/test/{0_real,1_fake} and Chameleon.zip.
    - 06_MMTD-Set: one JSON file.
    """
    import zipfile

    from eviground.data.fake_sources import make_diffseg_zip

    root = Path(datasets_root)
    rng = np.random.default_rng(seed)

    gi = root / "01_GenImage" / "Validation"
    for gen in FAKE_GENERATORS:
        for i in range(per_folder):
            _save(_generated(rng, 320), gi / gen / "val" / "ai" / f"{i:03d}_{gen}.PNG", "PNG")
            h = 200 if i == 0 else 300
            real = _real(rng, 400)[:h, :]
            _save(real, gi / gen / "val" / "nature" / f"ILSVRC_{i:05d}.JPEG", "JPEG",
                  quality=96, subsampling=2)  # fmt: skip
    (gi / "_DONE").write_text("vtphatt2/GenImage-Validation")

    ds = root / "02_DiffSeg30k"
    for split in FAKE_MASK_VALUES:  # the published zips, as ensure_data stores them
        make_diffseg_zip(ds / f"{split}.zip", split, per_folder, seed=seed)
        # what the retired downloader left: empty split folders and header-only manifests
        (ds / split / "images").mkdir(parents=True, exist_ok=True)
        (ds / split / "masks").mkdir(parents=True, exist_ok=True)
        (ds / f"{split}_manifest.csv").write_text("image,mask\n")

    sid = root / "03_SID-Set"
    labels = {0: "real", 1: "full_synthetic", 2: "tampered"}
    for split in ("train", "validation"):
        rows = []
        for lab, name in labels.items():
            for i in range(per_folder):
                iid = f"{name}_{i:04d}"
                img_rel = f"{split}/{name}/images/{iid}.png"
                _save(_real(rng, 288), sid / img_rel, "PNG")
                mask_rel = ""
                if lab == 2:
                    mask = np.zeros((288, 288), dtype=np.uint8)
                    mask[40:120, 40:120] = 255
                    mask_rel = f"{split}/{name}/masks/{iid}_mask.png"
                    _save(mask, sid / mask_rel, "PNG")
                rows.append([img_rel, mask_rel, lab, name])
        with open(sid / f"{split}_manifest.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "mask", "label", "label_name"])
            writer.writerows(rows)

    br = root / "04_BR-Gen"
    br.mkdir(parents=True, exist_ok=True)
    for part in ("BR-Gen.zip.001", "BR-Gen.zip.002"):
        (br / part).write_bytes(rng.bytes(1024))

    ch = root / "05_Chameleon" / "test"
    with zipfile.ZipFile(root / "Chameleon.zip", "w") as zf:
        for cls, n in (("0_real", per_folder), ("1_fake", per_folder - 1)):
            for i in range(n):
                path = ch / cls / f"{i:04d}.jpg"
                _save(_real(rng, 300), path, "JPEG", quality=90)
                zf.write(path, f"test/{cls}/{i:04d}.jpg")

    mm = root / "06_MMTD-Set"
    mm.mkdir(parents=True, exist_ok=True)
    (mm / "annotations.json").write_text('[{"image": "x.jpg", "text": "example"}]')
    return root
