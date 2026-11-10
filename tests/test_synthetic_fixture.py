"""The synthetic dataset that every later test builds on."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from eviground.data.synthetic import make_synthetic_dataset


def _load(root: Path, rel: str) -> np.ndarray:
    return np.asarray(Image.open(root / rel))


def test_counts_labels_and_shapes(synthetic_dataset: dict) -> None:
    records, root = synthetic_dataset["records"], synthetic_dataset["root"]
    size, per_class = synthetic_dataset["size"], synthetic_dataset["per_class"]
    assert len(records) == 3 * per_class
    for label in (0, 1, 2):
        assert sum(r["label"] == label for r in records) == per_class
    for record in records:
        assert _load(root, record["path"]).shape == (size, size, 3)
    assert (root / "manifest.csv").is_file()


def test_only_edited_images_have_masks(synthetic_dataset: dict) -> None:
    for record in synthetic_dataset["records"]:
        assert bool(record["mask"]) == (record["label"] == 2)


def test_edit_is_exactly_inside_the_mask(synthetic_dataset: dict) -> None:
    root, size = synthetic_dataset["root"], synthetic_dataset["size"]
    for record in (r for r in synthetic_dataset["records"] if r["label"] == 2):
        edited = _load(root, record["path"]).astype(int)
        base = _load(root, record["base"]).astype(int)
        mask = _load(root, record["mask"])
        assert set(np.unique(mask)) == {0, 255}
        assert (mask > 0).sum() == (size // 4) ** 2
        changed = np.abs(edited - base).sum(axis=-1) > 0
        assert not changed[mask == 0].any(), "pixels outside the mask must be untouched"
        assert changed[mask > 0].mean() > 0.5, "the masked square must actually differ"


def test_no_duplicate_images(synthetic_dataset: dict) -> None:
    root = synthetic_dataset["root"]
    digests = [
        hashlib.sha256((root / r["path"]).read_bytes()).hexdigest()
        for r in synthetic_dataset["records"]
    ]
    assert len(set(digests)) == len(digests)


def test_same_seed_gives_identical_bytes(tmp_path: Path) -> None:
    a = make_synthetic_dataset(tmp_path / "a", per_class=3, size=64, seed=7)
    b = make_synthetic_dataset(tmp_path / "b", per_class=3, size=64, seed=7)
    c = make_synthetic_dataset(tmp_path / "c", per_class=3, size=64, seed=8)
    assert a == b
    for record in a:
        same = (tmp_path / "a" / record["path"]).read_bytes()
        assert same == (tmp_path / "b" / record["path"]).read_bytes()
    assert any(
        (tmp_path / "a" / r["path"]).read_bytes() != (tmp_path / "c" / r["path"]).read_bytes()
        for r in c
    )
