"""Colab asks for the right GPU when a notebook is opened from GitHub (Dhanush, 27 Nov).

Without `accelerator` / `colab.gpuType` in the metadata, Colab reuses the last runtime type."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize(
    ("name", "gpu"),
    [("17_phase_c_evaluation", "L4"), ("17b_t4_parity", "T4"), ("18_cp6_reports", "T4")],
)
def test_gpu_notebooks_ask_colab_for_their_gpu(name, gpu):
    meta = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())["metadata"]
    assert meta["accelerator"] == "GPU"
    assert meta["colab"]["gpuType"] == gpu
