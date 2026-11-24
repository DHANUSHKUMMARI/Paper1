from __future__ import annotations

import json
import sys
import time
import types

import pytest

from eviground.utils import session


def test_preflight_prints_and_returns_the_balance(tmp_path, capsys):
    ckpt = tmp_path / "best.pt"
    ckpt.write_text("x")
    report = session.preflight(
        {"experiment": "eviground_ctx", "result": "gate_ctx.json"},
        paths={"checkpoint": ckpt},
        est_cu=0.3,
        cu_left=45.0,
        expect={"experiment": "eviground_ctx"},
    )
    out = capsys.readouterr().out
    assert "eviground_ctx" in out
    assert "CU left after: 44.7" in out
    assert report["cu_left_after"] == 44.7


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"paths": {"checkpoint": "/nope/best.pt"}}, "missing checkpoint"),
        ({"expect": {"experiment": "eviground_ctx"}}, "expected 'eviground_ctx'"),
        ({"cu_left": 0.1}, "exceeds"),
    ],
)
def test_preflight_refuses(kwargs, match):
    with pytest.raises(session.PreflightError, match=match):
        session.preflight({"experiment": None}, est_cu=0.3, **kwargs)


def test_finish_writes_compute_json(tmp_path):
    info = session.finish(tmp_path, time.time() - 1800, 2.0, extra={"git_commit": "abc"})
    saved = json.loads((tmp_path / "compute.json").read_text())
    assert saved == info
    assert saved["git_commit"] == "abc"
    assert abs(saved["compute_units"] - 1.0) < 0.01


def test_release_is_a_no_op_in_dry_runs_and_outside_colab(monkeypatch):
    assert session.release(dry=True) == "dry"
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert session.release(dry=False) == "not_colab"


def test_release_unassigns_in_colab(monkeypatch):
    calls = []
    fake = types.ModuleType("google.colab")
    fake.runtime = types.SimpleNamespace(unassign=lambda: calls.append(1))
    monkeypatch.setitem(sys.modules, "google.colab", fake)
    assert session.release(dry=False) == "unassigned"
    assert calls == [1]


def test_gpu_type_and_rate():
    assert session.gpu_type("NVIDIA L4") == "L4"
    assert session.gpu_type("Tesla T4") == "T4"
    assert session.gpu_type(None) == "cpu"
    assert session.cu_rate("NVIDIA L4") == 2.2
    assert session.cu_rate("Tesla T4") == 1.28
    assert session.cu_rate("Some new GPU") is None


def test_finish_takes_the_rate_of_the_gpu_it_ran_on(tmp_path):
    info = session.finish(tmp_path, time.time() - 3600, 2.2, gpu="Tesla T4")
    assert info["gpu_type"] == "T4"
    assert info["cu_per_hour"] == 1.28
    assert abs(info["cu_estimate"] - 1.28) < 0.01
    assert info["cu_source"] == "rate x wall time"


def test_the_balance_drop_wins_over_the_rate(tmp_path):
    info = session.finish(
        tmp_path, time.time() - 3600, gpu="NVIDIA L4", balance_start=28.19, balance_end=26.48
    )
    assert info["cu_measured"] == 1.71
    assert info["cu"] == 1.71
    assert info["cu_source"] == "balance"
    assert abs(info["cu_estimate"] - 2.2) < 0.01


def test_with_balance_after_the_run(tmp_path):
    info = session.finish(tmp_path, time.time() - 3600, gpu="NVIDIA L4", balance_start=10.0)
    out = session.with_balance(info, 7.5)
    assert out["cu"] == 2.5
    assert out["compute_units"] == 2.5
    assert out["cu_source"] == "balance"
    assert info["cu_source"] == "rate x wall time"  # the input is not changed


def test_with_balance_needs_the_start_balance(tmp_path):
    info = session.finish(tmp_path, time.time() - 60, 2.2)
    with pytest.raises(ValueError, match="cu_balance_start"):
        session.with_balance(info, 7.5)


def test_an_unknown_gpu_has_no_estimate(tmp_path):
    info = session.finish(tmp_path, time.time() - 60, gpu="Some new GPU")
    assert info["cu_estimate"] is None
    assert info["cu"] is None
