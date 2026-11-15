"""Milestone 3, PR C: trainer, checkpoints, selection metrics (CPU, tiny CLIP, fake world)."""

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
from eviground.data.datasets import ForensicDataset  # noqa: E402
from eviground.eval import metrics  # noqa: E402
from eviground.train import checkpoint as ckpt  # noqa: E402
from eviground.train.trainer import KilledForTest, Trainer, _collate  # noqa: E402


def _cfg(**over):
    cfg = load_config("debug", [f"{k}={v}" for k, v in over.items()])
    return cfg


def _trainer(packed, run_dir, **kw):  # noqa: F811
    return Trainer(_cfg(), packed["out"], run_dir, seed=42, log=None, **kw)


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ runs


@pytest.fixture(scope="module")
def full_run(packed, tmp_path_factory):  # noqa: F811
    run_dir = tmp_path_factory.mktemp("full")
    tr = _trainer(packed, run_dir)
    out = tr.fit()
    return tr, out, run_dir


def test_two_epoch_smoke_run_writes_the_run_folder(full_run):
    tr, out, run_dir = full_run
    assert out["epoch"] == 2 and out["global_step"] == 2 * tr.steps_per_epoch
    names = {p.name for p in run_dir.iterdir()}
    assert {"config.yaml", "metrics.csv", "steps.csv", "compute.json", "last.pt"} <= names
    assert "best.pt" in names
    assert len(list(run_dir.glob("step_*.pt"))) == 2  # rotation keeps 2
    rows = _rows(run_dir / "metrics.csv")
    assert [r["epoch"] for r in rows] == ["0", "1"]
    for col in (
        "train_total",
        "train_bce",
        "train_dice",
        "val_selection",
        "val_macro_auc",
        "val_mask_f1",
        "val_mask_f1_pooled",
        "val_mask_iou_pooled",
        "val_real_any_pred",
        "val_real_pred_area",
        "val_acc_genimage",
        "index_sha256",
        "manifest_sha256",
        "mask_rule",
        "clip_revision",
    ):
        assert col in rows[0], col
    assert rows[0]["index_sha256"] == tr.versions["index_sha256"]
    assert rows[0]["mask_rule"].startswith("diffseg30k")
    steps = _rows(run_dir / "steps.csv")
    assert [int(r["global_step"]) for r in steps] == list(range(1, out["global_step"] + 1))
    compute = json.loads((run_dir / "compute.json").read_text())
    assert compute["global_step"] == out["global_step"] and "compute_units" in compute


def test_checkpoint_holds_buffers_rng_and_versions(full_run):
    tr, _out, run_dir = full_run
    state = ckpt.load(run_dir / "last.pt")
    keys = set(state["model"])
    assert any(k.endswith("running_mean") for k in keys)  # EfficientNet BatchNorm
    assert any(k.endswith("num_batches_tracked") for k in keys)
    assert any("lora_" in k for k in keys)
    assert not any(ckpt.is_frozen_clip(k) for k in keys)
    assert {"python", "numpy", "torch"} <= set(state["rng"])
    assert state["clip_revision"] == "tiny" and state["clip_sha256"] == tr.clip_sha256
    assert state["data"]["index_sha256"] == tr.versions["index_sha256"]
    assert state["data"]["mask_rule"] == tr.versions["mask_rule"]


def test_resumed_model_gives_identical_eval_outputs(full_run, packed, tmp_path):  # noqa: F811
    tr, _out, run_dir = full_run
    # same seed: the same (random tiny) frozen CLIP; the trained weights come from the file
    other = Trainer(_cfg(), packed["out"], tmp_path, seed=42, log=None, resume=None)
    batch = _collate([tr.val_ds[k] for k in range(len(tr.val_ds))])
    tr.model.eval()
    other.model.eval()
    with torch.no_grad():
        want = tr.model(batch["image"])
        before = other.model(batch["image"])
    assert not torch.equal(want["logits"], before["logits"])  # untrained differs
    other.resume(run_dir / "last.pt")
    other.model.eval()
    with torch.no_grad():
        got = other.model(batch["image"])
    for k in ("logits", "mask_logits", "alpha", "gate"):
        assert torch.equal(want[k], got[k]), k


def test_kill_and_resume_gives_the_same_steps(full_run, packed, tmp_path):  # noqa: F811
    _tr, out, run_dir = full_run
    tr = _trainer(packed, tmp_path, kill_at_step=3)  # checkpoints every 2 steps (debug)
    with pytest.raises(KilledForTest):
        tr.fit()
    again = _trainer(packed, tmp_path)
    assert again.state["global_step"] == 2 and again.state["step_in_epoch"] == 2
    res = again.fit()
    assert res["global_step"] == out["global_step"]
    full = {r["global_step"]: r for r in _rows(run_dir / "steps.csv")}
    resumed = {r["global_step"]: r for r in _rows(tmp_path / "steps.csv")}  # last row wins
    for step in range(3, out["global_step"] + 1):
        a, b = full[str(step)], resumed[str(step)]
        assert a["first_uid"] == b["first_uid"]
        la, lb = float(a["loss_total"]), float(b["loss_total"])
        assert abs(la - lb) <= 0.01 * abs(la), (step, la, lb)
    # the epoch's train loss covers all its micro-batches, also across the kill
    full_m, cut_m = _rows(run_dir / "metrics.csv"), _rows(tmp_path / "metrics.csv")
    for x, y in zip(full_m, cut_m, strict=True):
        assert float(x["train_total"]) == pytest.approx(float(y["train_total"]), rel=0.01)


def test_an_interrupted_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    ckpt.save(path, {"x": 1})

    def broken(_obj, fh):
        fh.write(b"half a checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError):
        ckpt.save(path, {"x": 2})
    monkeypatch.undo()
    assert ckpt.load(path) == {"x": 1}


def test_rotation_keeps_the_newest_step_files(tmp_path):
    for s in (5, 10, 15, 20):
        (tmp_path / f"step_{s:07d}.pt").write_bytes(b"")
    gone = ckpt.rotate(tmp_path, keep=2)
    assert sorted(p.name for p in gone) == ["step_0000005.pt", "step_0000010.pt"]
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [
        "step_0000015.pt",
        "step_0000020.pt",
    ]


def test_a_changed_frozen_clip_or_revision_stops_the_resume(full_run, packed, tmp_path):  # noqa: F811
    _tr, _out, run_dir = full_run
    state = ckpt.load(run_dir / "last.pt")
    other = Trainer(_cfg(), packed["out"], tmp_path, seed=42, log=None, resume=None)
    with pytest.raises(ckpt.CheckpointMismatch, match="revision"):
        ckpt.restore_model(
            other.model, state, clip_revision="abc123", clip_sha256=other.clip_sha256
        )
    with torch.no_grad():
        next(p for n, p in other.model.named_parameters() if ckpt.is_frozen_clip(n)).add_(1e-3)
    changed = ckpt.frozen_clip_sha256(other.model)
    assert changed != other.clip_sha256
    with pytest.raises(ckpt.CheckpointMismatch, match="frozen CLIP"):
        ckpt.restore_model(other.model, state, clip_revision="tiny", clip_sha256=changed)


def test_a_different_shard_index_stops_the_resume(full_run, packed, tmp_path):  # noqa: F811
    _tr, _out, run_dir = full_run
    other = Trainer(_cfg(), packed["out"], tmp_path, seed=42, log=None, resume=None)
    other.versions = {**other.versions, "index_sha256": "0" * 64}
    with pytest.raises(ckpt.CheckpointMismatch, match="shard index"):
        other.resume(run_dir / "last.pt")


def test_a_real_backbone_needs_a_pinned_revision(packed, tmp_path):  # noqa: F811
    cfg = _cfg()
    cfg.model.semantic.tiny = None
    cfg.model.semantic.revision = None
    with pytest.raises(ValueError, match="revision"):
        Trainer(cfg, packed["out"], tmp_path, log=None)


# ------------------------------------------------------------------ augmentation RNG (fix 3)


def test_items_are_the_same_with_0_and_2_workers_and_after_a_resume(packed):  # noqa: F811
    from torch.utils.data import DataLoader, Subset

    ds = ForensicDataset(packed["out"], "train", n_per_class=8, seed=42)
    ds.set_epoch(1)
    k0 = 5

    def batches(workers: int, d: ForensicDataset):
        dl = DataLoader(
            Subset(d, range(k0, len(d))), batch_size=4, num_workers=workers, collate_fn=_collate
        )
        return [(b["uid"], b["image"], b["mask"]) for b in dl]

    a, b = batches(0, ds), batches(2, ds)
    fresh = ForensicDataset(packed["out"], "train", n_per_class=8, seed=42)
    fresh.set_epoch(1)  # a restarted process
    c = batches(0, fresh)
    for (ua, ia, ma), (ub, ib, mb), (uc, ic, mc) in zip(a, b, c, strict=True):
        assert ua == ub == uc
        assert torch.equal(ia, ib) and torch.equal(ia, ic)
        assert torch.equal(ma, mb) and torch.equal(ma, mc)


# ------------------------------------------------------------------ overfit path


def test_overfit_runs_with_augmentation_and_dropout_off(packed, tmp_path):  # noqa: F811
    tr = _trainer(packed, tmp_path, resume=None)
    res = tr.overfit()
    assert res["images"] == 8 and set(res["by_class"]) == {"real", "generated", "edited"}
    assert res["history"] and res["history"][-1]["step"] <= 4
    drops = [m.p for m in tr.model.modules() if isinstance(m, torch.nn.Dropout)]
    assert drops and all(p == 0.0 for p in drops)


# ------------------------------------------------------------------ metrics (fix 4)


def test_binary_auc_matches_the_pair_count():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.4, 0.4, 0.1, 0.8, 0.7]
    pos = [a for a, ya in zip(s, y, strict=True) if ya]
    neg = [b for b, yb in zip(s, y, strict=True) if not yb]
    pairs = [(a, b) for a in pos for b in neg]
    want = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
    assert metrics.binary_auc(y, s) == pytest.approx(want)
    assert metrics.binary_auc([1, 1], [0.2, 0.3]) is None


def test_mask_f1_is_the_mean_of_per_image_f1_and_real_false_positives():
    t = np.zeros((3, 4, 4), dtype=bool)
    p = np.zeros_like(t)
    t[0, :2] = True  # 8 pixels, predicted exactly: F1 1
    p[0, :2] = True
    t[1, :, :1] = True  # 4 pixels, predicted 2 of them + 2 wrong: F1 0.5
    p[1, :2, :1] = True
    p[1, 3, 2:] = True
    p[2, 0, 0] = True  # a real crop with one predicted pixel
    s = metrics.mask_scores(p, t, [2, 2, 0])
    assert s["mask_f1"] == pytest.approx(0.75)
    assert s["mask_f1_pooled"] == pytest.approx(2 * 10 / (2 * 10 + 2 + 2))
    assert s["mask_iou_pooled"] == pytest.approx(10 / 14)
    assert s["real_any_pred"] == 1.0 and s["real_pred_area"] == pytest.approx(1 / 16)


def test_selection_score_for_both_profiles():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.1, 0.6],
        ]
    )
    uni = metrics.selection_score("unified", labels, probs, mask_f1=0.5)
    assert uni["macro_auc"] == 1.0 and uni["selection"] == pytest.approx(0.75)
    assert metrics.selection_score("unified", labels, probs, None)["selection"] == 1.0
    det = metrics.selection_score("detection", labels, probs, None)
    assert det["selection"] == det["det_auc"] == 1.0


def test_a_resume_with_another_profile_or_seed_stops(full_run, packed, tmp_path):  # noqa: F811
    _tr, _out, run_dir = full_run
    det = Trainer(
        _cfg(), packed["out"], tmp_path / "d", seed=42, profile="detection", log=None, resume=None
    )
    with pytest.raises(ckpt.CheckpointMismatch, match="profile"):
        det.resume(run_dir / "last.pt")
    state = ckpt.load(run_dir / "last.pt")
    state["seed"] = 123
    ckpt.save(tmp_path / "other_seed.pt", state)
    same = Trainer(_cfg(), packed["out"], tmp_path / "s", seed=42, log=None, resume=None)
    with pytest.raises(ckpt.CheckpointMismatch, match="seed"):
        same.resume(tmp_path / "other_seed.pt")


def test_a_short_last_accumulation_group_is_scaled_by_its_size(packed, tmp_path):  # noqa: F811
    tr = Trainer(_cfg(**{"train.grad_accum": 4}), packed["out"], tmp_path, log=None, resume=None)
    assert tr.micro_per_epoch == 6 and tr.steps_per_epoch == 2
    sizes = []
    for k in range(tr.micro_per_epoch):
        tr.state["step_in_epoch"] = k
        sizes.append(tr._group_size())
    assert sizes == [4, 4, 4, 4, 2, 2]


def test_max_steps_then_resume_keeps_the_epoch_loss(full_run, packed, tmp_path):  # noqa: F811
    _tr, _out, run_dir = full_run
    first = _trainer(packed, tmp_path)
    first.fit(max_steps=3)  # stops inside epoch 0 (6 steps)
    again = _trainer(packed, tmp_path)
    assert again.state["epoch_micro"] == 3
    again.fit()
    want, got = _rows(run_dir / "metrics.csv"), _rows(tmp_path / "metrics.csv")
    for x, y in zip(want, got, strict=True):
        assert float(x["train_total"]) == pytest.approx(float(y["train_total"]), rel=0.01)
