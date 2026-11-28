"""Colab asks for the right GPU when a notebook is opened from GitHub (Dhanush, 27 Nov).

Without `accelerator` / `colab.gpuType` in the metadata, Colab reuses the last runtime type."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize(
    ("name", "gpu"),
    [
        ("17_phase_c_evaluation", "L4"),
        ("17b_t4_parity", "T4"),
        ("18_cp6_reports", "T4"),
        ("19_cp8_generation", "T4"),
    ],
)
def test_gpu_notebooks_ask_colab_for_their_gpu(name, gpu):
    meta = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())["metadata"]
    assert meta["accelerator"] == "GPU"
    assert meta["colab"]["gpuType"] == gpu


def _source(name):
    nb = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_phase_c_start_balance_is_a_required_setting():
    src = _source("17_phase_c_evaluation")
    assert "PHASE_C_BALANCE_START = None" in src  # no fixed default (it was 23.43)
    assert "assert isinstance(PHASE_C_BALANCE_START, (int, float))" in src
    assert "phasec.session_cap(CU_CAP, PHASE_C_BALANCE_START, CU_BALANCE_START)" in src
    assert "cap this session (effective)" in src


def test_the_cp8_block_skips_cleanly_without_its_inputs():
    src = _source("17_phase_c_evaluation")
    assert "if not testcache.is_ready(CACHE, CP8_TAR):" in src
    assert "'status': 'skipped', 'reason': f'inputs missing" in src


def test_cp8_generation_asks_for_high_ram_and_stops_on_repeated_failures():
    meta = json.loads((NOTEBOOKS / "19_cp8_generation.ipynb").read_text())["metadata"]
    assert meta["colab"]["machine_shape"] == "hm"  # Kolors with CPU offload: ~18 GB RAM
    src = _source("19_cp8_generation")
    assert "MAX_FAILS = 3" in src
    assert "ce.consecutive_failures(" in src
    assert ">= MAX_FAILS" in src



def test_cp8_install_pins_a_python_313_stack_and_shows_errors():
    """transformers 4.42.4 needs tokenizers 0.19, which does not install on Python 3.13 (28 Nov)."""
    src = _source("19_cp8_generation")
    assert "'transformers==4.45.2', 'tokenizers==0.20.3'" in src
    assert "transformers==4.42.4" not in src
    assert "'-q', *KOLORS_PINS" not in src  # pip output is captured and its tail shown on failure
    assert "[-30:]" in src


def test_cp8_install_removes_colabs_peft_before_diffusers_is_used():
    """Colab's peft needs accelerate >= 1.x; diffusers 0.28.2 imports it if present (28 Nov)."""
    src = _source("19_cp8_generation")
    assert "'uninstall', '-y', 'peft'" in src
    assert src.index("*KOLORS_PINS]") < src.index("'uninstall', '-y', 'peft'")
    assert src.index("'uninstall', '-y', 'peft'") < src.index("inpaint.KolorsInpainter(WEIGHTS)")
    assert "'diffusers' in sys.modules" in src


def test_cp8_install_restores_the_hub_pin_after_the_repo_install():
    """`pip install -e REPO` upgrades huggingface-hub (eviground needs >= 0.24) on every Run all;
    diffusers 0.28.2 needs cached_download, removed in hub 0.26 (28 Nov)."""
    src = _source("19_cp8_generation")
    assert "_version('huggingface-hub') != '0.23.4'" in src
    assert src.index("'-e', REPO]") < src.index("_version('huggingface-hub') != '0.23.4'")
    assert src.index("_version('huggingface-hub') != '0.23.4'") < src.index("*KOLORS_PINS]")
    assert "'huggingface_hub' in sys.modules" in src  # reinstall after import: restart
