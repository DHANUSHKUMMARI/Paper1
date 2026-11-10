"""Shared fixtures. Everything here is synthetic and CPU-only: no downloads, no real data."""

from __future__ import annotations

from pathlib import Path

import pytest

from eviground.data.synthetic import make_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_CLASS = 12
IMAGE_SIZE = 128
SEED = 42


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """12 images per class at 128x128; edited images carry a pasted square and its mask."""
    root = tmp_path_factory.mktemp("synthetic")
    records = make_synthetic_dataset(root, per_class=PER_CLASS, size=IMAGE_SIZE, seed=SEED)
    return {"root": root, "records": records, "per_class": PER_CLASS, "size": IMAGE_SIZE}


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the project at a temp 'Drive' and clear anything that could leak in."""
    for name in ("EVIGROUND_DRY_RUN", "HF_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    drive = tmp_path / "drive" / "Paper 1"
    monkeypatch.setenv("EVIGROUND_DRIVE_ROOT", str(drive))
    monkeypatch.setenv("EVIGROUND_LOCAL_DATA", str(tmp_path / "data"))
    return drive
