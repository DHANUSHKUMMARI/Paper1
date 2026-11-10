"""Configuration loading and composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from eviground.config import load_config, save_config


def test_base_config() -> None:
    cfg = load_config("base")
    assert list(cfg.seeds) == [42, 123, 2026]
    assert cfg.seed == 42
    assert cfg.deterministic is True
    assert list(cfg.data.balance) == [1, 1, 1]
    assert cfg.checkpoint.every_steps == 500


def test_debug_config_composes_base_and_wins() -> None:
    cfg = load_config("debug")
    assert "defaults" not in cfg
    assert list(cfg.seeds) == [42, 123, 2026]  # inherited from base
    assert cfg.data.crop == 128  # overridden
    assert cfg.data.clip_size == 224  # inherited
    assert cfg.model.semantic.pretrained is False and cfg.model.trace.pretrained is False
    assert cfg.compute.cu_per_hour == 0.0


def test_overrides() -> None:
    cfg = load_config("debug", overrides=["seed=2026", "train.batch_size=2"])
    assert cfg.seed == 2026 and cfg.train.batch_size == 2


def test_configs_contain_no_machine_paths(repo_root: Path) -> None:
    for path in (repo_root / "configs").rglob("*.yaml"):
        text = path.read_text()
        assert "/content/" not in text and "MyDrive" not in text, f"hard-coded path in {path}"


def test_missing_and_circular_configs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
    (tmp_path / "a.yaml").write_text("defaults: [b]\nx: 1\n")
    (tmp_path / "b.yaml").write_text("defaults: [a]\ny: 2\n")
    with pytest.raises(ValueError, match="circular"):
        load_config("a", config_dir=tmp_path)


def test_save_roundtrip(tmp_path: Path) -> None:
    cfg = load_config("debug")
    save_config(cfg, tmp_path / "run" / "config.yaml")
    again = load_config(tmp_path / "run" / "config.yaml")
    assert again == cfg
