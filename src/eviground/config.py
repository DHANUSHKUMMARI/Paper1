"""YAML configuration with a small ``defaults:`` list for composition (OmegaConf).

Example: ``configs/debug.yaml`` starts with ``defaults: [base]``; files listed there are
merged first, in order, and the file's own keys win. Paths never live in configs; they
come from :mod:`eviground.env`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eviground.env import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"


def _resolve(name: str | Path, config_dir: Path) -> Path:
    path = Path(name)
    if path.suffix not in {".yaml", ".yml"}:
        path = path.with_name(path.name + ".yaml")
    if not path.is_absolute():
        path = config_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return path


def _load(path: Path, config_dir: Path, stack: tuple[Path, ...]) -> DictConfig:
    if path in stack:
        chain = " -> ".join(p.name for p in (*stack, path))
        raise ValueError(f"circular config defaults: {chain}")
    own = OmegaConf.load(path)
    if not isinstance(own, DictConfig):
        raise TypeError(f"config must be a mapping at the top level: {path}")
    defaults = own.pop("defaults", None) or []
    merged = OmegaConf.create({})
    for parent in defaults:
        parent_cfg = _load(_resolve(parent, config_dir), config_dir, (*stack, path))
        merged = OmegaConf.merge(merged, parent_cfg)
    return OmegaConf.merge(merged, own)  # type: ignore[return-value]


def load_config(
    name: str | Path,
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> DictConfig:
    """Load ``configs/<name>.yaml`` with its defaults, then apply ``key=value`` overrides."""
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = _load(_resolve(name, config_dir), config_dir, ())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: Path) -> None:
    """Write the fully resolved config (one per run folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
