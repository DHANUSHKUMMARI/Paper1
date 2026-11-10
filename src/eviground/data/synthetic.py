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
            {"path": rel, "label": 2, "mask": mask_rel, "base": base_rel,
             "source": "synthetic/edited"}  # fmt: skip
        )

    with open(root / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return records
