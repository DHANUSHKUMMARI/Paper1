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
