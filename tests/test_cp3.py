"""CP3 (training health): new metrics, trainer additions and the criteria evaluator."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("timm")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from test_data_pipeline import packed  # noqa: E402, F401  (module fixture: fake-world shards)

from eviground.config import load_config  # noqa: E402
from eviground.eval import cp3, metrics  # noqa: E402
from eviground.train import trainer as trainer_mod  # noqa: E402
from eviground.train.trainer import NonFiniteLoss, Trainer  # noqa: E402

# ------------------------------------------------------------------ metrics


def test_binary_miou_and_generated_crops_by_hand():
    t = np.zeros((3, 2, 4), dtype=bool)
    p = np.zeros_like(t)
    t[0, 0, :2] = True  # edited: 2 target pixels, predicted 1 of them + 1 wrong
    p[0, 0, 0] = True
    p[0, 1, 3] = True
    p[1, 0, 0] = True  # generated crop with one predicted pixel (of 8)
    labels = [2, 1, 0]  # edited, generated, real (nothing predicted)
    s = metrics.mask_scores(p, t, labels)
    # edited crop: TP 1, FP 1, FN 1, TN 5 -> IoU edited 1/3, IoU background 5/7
    assert s["mask_iou_pooled"] == pytest.approx(1 / 3)
    assert s["mask_miou_binary"] == pytest.approx((1 / 3 + 5 / 7) / 2)
    assert s["gen_any_pred"] == 1.0
    assert s["gen_pred_area"] == pytest.approx(1 / 8)
    assert s["real_any_pred"] == 0.0
    assert s["real_pred_area"] == 0.0


def test_per_source_mask_report():
    pred = np.zeros((4, 2, 2), dtype=bool)
    pred[1, 0, 0] = True
    labels = [0, 0, 0, 2]
    called = [0, 2, 0, 2]
    sources = ["coco", "diffseg30k", "diffseg30k", "diffseg30k"]
    r = metrics.per_source_mask_report(pred, labels, called, sources)
    assert r["coco_real_called_edited"] == 0.0
    assert r["diffseg30k_real_called_edited"] == 0.5
    assert r["diffseg30k_real_any_pred"] == 0.5
    assert r["diffseg30k_real_pred_area"] == pytest.approx(0.125)
    assert r["diffseg30k_edited_called_edited"] == 1.0


def test_detection_accuracy_uses_one_minus_p_real():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1, 0], [0.4, 0.1, 0.5], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1]])
    s = metrics.selection_score("detection", labels, probs, None)
    # 1 - P(real) = 0.1, 0.6, 0.7, 0.4 -> predicted generated for #2, #3 -> 2 of 4 right
    assert s["det_acc"] == 0.5


def test_plateau_gain():
    assert metrics.plateau_gain([0.5, 0.6, 0.7]) is None
    assert metrics.plateau_gain([0.5, 0.6, 0.7, 0.701, 0.702, 0.703, 0.704]) == pytest.approx(0.004)


# ------------------------------------------------------------------ trainer


def _cfg(**over):
    return load_config("debug", [f"{k}={v}" for k, v in over.items()])


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_a_run_logs_train_eval_per_source_and_grad_norm(packed, tmp_path):  # noqa: F811
    tr = Trainer(_cfg(), packed["out"], tmp_path, seed=42, log=None)
    out = tr.fit()
    row = _rows(tmp_path / "metrics.csv")[0]
    for col in (
        "val_mask_miou_binary",
        "val_gen_any_pred",
        "val_gen_pred_area",
        "val_coco_real_called_edited",
        "val_genimage_real_any_pred",
        "train_eval_selection",
        "train_eval_mask_f1",
        "best_epoch",
    ):
        assert col in row, col
    steps = _rows(tmp_path / "steps.csv")
    assert all(np.isfinite(float(r["grad_norm"])) for r in steps)
    assert "still_improving_at_end" in out
    assert out["best_epoch"] is not None


def test_overfit_runs_until_both_targets_and_records_mask_f1(packed, tmp_path, monkeypatch):  # noqa: F811
    tr = Trainer(_cfg(**{"train.overfit.mask_steps": 10}), packed["out"], tmp_path, log=None)
    script = iter([(0.5, 0.1), (1.0, 0.3), (1.0, 0.85), (1.0, 0.9), (1.0, 0.95)])

    def point(ds, step, loss):
        acc, f1 = next(script)
        return {"step": step, "loss": loss, "acc": acc, "mask_f1": f1}

    monkeypatch.setattr(tr, "_overfit_point", point)
    res = tr.overfit()  # eval every 2 steps (debug)
    assert res["acc_reached_at_step"] == 4  # within steps=4
    assert res["mask_f1_reached_at_step"] == 6
    assert res["acc_pass"] is True
    assert res["mask_pass"] is True
    assert [h["step"] for h in res["history"]] == [2, 4, 6]  # stopped once both held
    assert res["final_mask_f1"] == 0.85


def test_overfit_fails_when_accuracy_comes_after_its_window(packed, tmp_path, monkeypatch):  # noqa: F811
    tr = Trainer(_cfg(), packed["out"], tmp_path, log=None)  # steps 4, mask_steps 6

    def point(ds, step, loss):
        return {"step": step, "loss": loss, "acc": 1.0 if step >= 6 else 0.5, "mask_f1": 0.9}

    monkeypatch.setattr(tr, "_overfit_point", point)
    res = tr.overfit()
    assert res["acc_reached_at_step"] == 6
    assert res["acc_pass"] is False
    assert res["reached_at_step"] is None


def test_a_non_finite_loss_stops_without_a_step(packed, tmp_path, monkeypatch):  # noqa: F811
    tr = Trainer(_cfg(), packed["out"], tmp_path, seed=42, log=None)
    real_loss = tr._loss
    calls = {"n": 0}

    def flaky(batch, train=True):
        out, loss = real_loss(batch, train)
        calls["n"] += 1
        if train and calls["n"] == 3:
            loss = {**loss, "bce": loss["bce"] * float("nan")}
        return out, loss

    monkeypatch.setattr(tr, "_loss", flaky)
    with pytest.raises(NonFiniteLoss, match="step 3"):
        tr.fit()
    assert tr.state["global_step"] == 2
    record = json.loads((tmp_path / "nonfinite.json").read_text())
    assert record["global_step"] == 3
    assert "bce" in record["non_finite"]
    assert (tmp_path / "last.pt").is_file()


def test_a_non_finite_gradient_stops_before_the_optimizer_step(packed, tmp_path, monkeypatch):  # noqa: F811
    tr = Trainer(_cfg(), packed["out"], tmp_path, seed=42, log=None)
    steps = {"n": 0}
    real_step = tr.optimizer.step

    def counting_step(*a, **k):
        steps["n"] += 1
        return real_step(*a, **k)

    monkeypatch.setattr(tr.optimizer, "step", counting_step)
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def flaky_clip(params, max_norm):
        norm = real_clip(params, max_norm)
        calls["n"] += 1
        return norm * float("nan") if calls["n"] == 3 else norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", flaky_clip)
    with pytest.raises(NonFiniteLoss, match="step 3"):
        tr.fit()
    assert steps["n"] == 2  # the optimizer never ran on the bad gradients
    assert tr.state["global_step"] == 2
    record = json.loads((tmp_path / "nonfinite.json").read_text())
    assert "grad_norm" in record["non_finite"]
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    assert saved["run"]["global_step"] == 2


def test_still_improving_at_end(packed, tmp_path):  # noqa: F811
    tr = Trainer(_cfg(), packed["out"], tmp_path, log=None, resume=None)
    tr.state.update(epoch=tr.epochs, stopped=False)
    tr.state["selection_history"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert tr.summary()["still_improving_at_end"] is True
    tr.state["selection_history"] = [0.1, 0.5, 0.5, 0.501, 0.502, 0.503]
    assert tr.summary()["still_improving_at_end"] is False
    assert trainer_mod.PLATEAU_GAIN == cp3.THRESHOLDS["S2b_plateau_gain"]


# ------------------------------------------------------------------ cp3 evaluator

GOOD = {
    "val_selection": 0.80,
    "val_macro_auc": 0.95,
    "val_auc_real": 0.94,
    "val_auc_generated": 0.97,
    "val_auc_edited": 0.93,
    "val_det_auc": 0.97,
    "val_det_acc": 0.96,
    "val_acc_genimage": 0.97,
    "val_mask_f1": 0.65,
    "val_mask_miou_binary": 0.71,
    "val_real_any_pred": 0.10,
    "val_real_pred_area": 0.004,
    "val_gen_any_pred": 0.12,
    "val_gen_pred_area": 0.005,
    "val_diffseg30k_real_called_edited": 0.05,
    "val_coco_real_called_edited": 0.03,
    "index_sha256": "abc",
}
OVERFIT_OK = {
    "acc_pass": True,
    "mask_pass": True,
    "acc_reached_at_step": 50,
    "mask_f1_reached_at_step": 400,
}


def _write_run(path: Path, epochs: list[dict], steps: list[dict] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for e in epochs for k in e})
    with open(path / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(epochs)
    steps = steps or [
        {"global_step": i, "loss_total": 1.0, "loss_bce": 0.5, "grad_norm": 0.7} for i in range(5)
    ]
    with open(path / "steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(steps[0]))
        w.writeheader()
        w.writerows(steps)
    return path


def _epochs(best: dict, n: int = 8, early: bool = True) -> list[dict]:
    """Selection rising to ``best`` at epoch 3, then flat (early stop at the end)."""
    out = []
    for ep in range(n):
        sel = min(0.3 + 0.17 * ep, best["val_selection"]) if ep < 3 else best["val_selection"]
        row = {**best, "epoch": ep, "val_selection": sel if ep != 3 else best["val_selection"]}
        row["train_total"] = 2.0 - 0.2 * min(ep, 3)
        row["bad_epochs"] = max(0, ep - 3) if early else 0
        if ep > 3:
            row["val_selection"] = best["val_selection"] - 0.01
        out.append(row)
    return out


def test_cp3_passes_a_healthy_unified_run(tmp_path):
    run = _write_run(tmp_path / "u", _epochs(GOOD))
    rep = cp3.evaluate_run(run, "unified", OVERFIT_OK)
    assert rep["verdict"] == "PASS", {k: v["status"] for k, v in rep["checks"].items()}
    assert rep["best_epoch"] == 3
    assert rep["checks"]["R2"]["mask_miou_binary"] == 0.71
    assert rep["vs_untrained_cp2"]["val_mask_f1"] == {"untrained": 0.2162, "best": 0.65}


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"val_macro_auc": 0.85}, "S2a"),
        ({"val_real_any_pred": 0.3}, "A2"),
        ({"val_real_pred_area": 0.02}, "A2"),
        ({"val_gen_any_pred": 0.25}, "A3"),
        ({"val_mask_f1": 0.4}, "A4"),
    ],
)
def test_cp3_fails_each_threshold(tmp_path, change, check):
    run = _write_run(tmp_path / "u", _epochs({**GOOD, **change}))
    rep = cp3.evaluate_run(run, "unified", OVERFIT_OK)
    assert rep["checks"][check]["status"] == "FAIL"
    assert rep["verdict"] == "FAIL"


def test_an_all_empty_mask_model_passes_a2_a3_but_fails_a4(tmp_path):
    empty = {
        **GOOD,
        "val_mask_f1": 0.0,
        "val_real_any_pred": 0.0,
        "val_real_pred_area": 0.0,
        "val_gen_any_pred": 0.0,
        "val_gen_pred_area": 0.0,
    }
    rep = cp3.evaluate_run(_write_run(tmp_path / "u", _epochs(empty)), "unified", OVERFIT_OK)
    assert rep["checks"]["A2"]["status"] == "PASS"
    assert rep["checks"]["A3"]["status"] == "PASS"
    assert rep["checks"]["A4"]["status"] == "FAIL"


def test_cp3_stop_and_ask_cases(tmp_path):
    weak = cp3.evaluate_run(
        _write_run(tmp_path / "a", _epochs({**GOOD, "val_auc_edited": 0.75})), "unified", OVERFIT_OK
    )
    assert weak["checks"]["S2a"]["status"] == "STOP_AND_ASK"
    assert weak["verdict"] == "STOP_AND_ASK"
    style = cp3.evaluate_run(
        _write_run(tmp_path / "b", _epochs({**GOOD, "val_diffseg30k_real_called_edited": 0.2})),
        "unified",
        OVERFIT_OK,
    )
    assert style["checks"]["R1"]["status"] == "STOP_AND_ASK"
    assert style["checks"]["R1"]["called_edited_gap"] == pytest.approx(0.17)
    rising = [
        {**GOOD, "epoch": e, "val_selection": 0.3 + 0.05 * e, "train_total": 2 - 0.1 * e}
        for e in range(8)
    ]
    for r in rising:
        r["bad_epochs"] = 0
    still = cp3.evaluate_run(_write_run(tmp_path / "c", rising), "unified", OVERFIT_OK)
    assert still["checks"]["S2b"]["status"] == "STOP_AND_ASK"


def test_cp3_s1_and_a1_failures(tmp_path):
    bad_steps = [{"global_step": 1, "loss_total": "nan", "grad_norm": 0.5}]
    run = _write_run(tmp_path / "u", _epochs(GOOD), bad_steps)
    rep = cp3.evaluate_run(run, "unified", OVERFIT_OK)
    assert rep["checks"]["S1"]["status"] == "FAIL"
    no_mask = cp3.evaluate_run(
        _write_run(tmp_path / "v", _epochs(GOOD)), "unified", {**OVERFIT_OK, "mask_pass": False}
    )
    assert no_mask["checks"]["A1"]["status"] == "FAIL"


def test_cp3_detection_run(tmp_path):
    run = _write_run(tmp_path / "d", _epochs(GOOD))
    rep = cp3.evaluate_run(run, "detection")
    assert rep["verdict"] == "PASS"
    assert set(rep["checks"]) == {"S1", "S2a", "S2b", "S3"}
    low = cp3.evaluate_run(
        _write_run(tmp_path / "e", _epochs({**GOOD, "val_det_acc": 0.9})), "detection"
    )
    assert low["checks"]["S3"]["status"] == "FAIL"


def test_cp3_report_round_trip(tmp_path):
    rep = cp3.evaluate_run(_write_run(tmp_path / "u", _epochs(GOOD)), "unified", OVERFIT_OK)
    path = cp3.write_report(rep, tmp_path / "out" / "cp3.json")
    assert json.loads(path.read_text())["verdict"] == "PASS"
