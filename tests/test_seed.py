"""Reproducibility: the same seed must give the same numbers everywhere."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from eviground.utils import DEFAULT_SEEDS, make_generator, seed_everything


def test_default_seeds() -> None:
    assert DEFAULT_SEEDS == (42, 123, 2026)


@pytest.mark.parametrize("seed", DEFAULT_SEEDS)
def test_python_and_numpy_are_reproducible(seed: int) -> None:
    seed_everything(seed)
    first = (random.random(), np.random.rand(4).tolist())
    seed_everything(seed)
    second = (random.random(), np.random.rand(4).tolist())
    assert first == second
    assert os.environ["PYTHONHASHSEED"] == str(seed)


def test_different_seeds_differ() -> None:
    seed_everything(42)
    a = np.random.rand(4).tolist()
    seed_everything(123)
    assert a != np.random.rand(4).tolist()


@pytest.mark.parametrize("bad", [-1, 1.5, "42", True, None])
def test_invalid_seed_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        seed_everything(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", DEFAULT_SEEDS)
def test_torch_is_reproducible(seed: int) -> None:
    torch = pytest.importorskip("torch")
    seed_everything(seed)
    first = torch.rand(4)
    layer_a = torch.nn.Linear(8, 4).weight.detach().clone()
    seed_everything(seed)
    assert torch.equal(first, torch.rand(4))
    assert torch.equal(layer_a, torch.nn.Linear(8, 4).weight.detach())


def test_deterministic_switches_are_set() -> None:
    torch = pytest.importorskip("torch")
    seed_everything(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()


def test_generator_is_seeded() -> None:
    torch = pytest.importorskip("torch")
    a = torch.randperm(10, generator=make_generator(42))
    b = torch.randperm(10, generator=make_generator(42))
    assert torch.equal(a, b)
