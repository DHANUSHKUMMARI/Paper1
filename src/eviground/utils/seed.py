"""Determinism switches, all in one place.

Every entry point calls :func:`seed_everything` once, before any data or model is built.
The paper reports mean and standard deviation over ``DEFAULT_SEEDS``.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SEEDS: tuple[int, int, int] = (42, 123, 2026)


def seed_everything(seed: int = DEFAULT_SEEDS[0], deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs) and return the seed.

    With ``deterministic=True`` cuDNN autotuning is disabled and PyTorch is asked for
    deterministic algorithms (warning, not failing, where an operation has none).
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        log.debug("torch not installed; seeded python and numpy only")
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Needed by cuBLAS for deterministic matrix products on GPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader workers: derive each worker's seed from torch's."""
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int):
    """A seeded ``torch.Generator`` for samplers and DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
