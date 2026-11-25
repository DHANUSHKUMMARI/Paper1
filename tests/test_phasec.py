"""PR Z: Phase C helpers (eval/phasec.py): the CU-capped blocks, items from staged rows, the
per-protocol summaries."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eviground.eval import phasec
from eviground.eval.tiled import mask_record


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_blocks_run_reuse_and_stop_at_the_cap(tmp_path):
    clock, log = Clock(0.0), {}
    calls = []

    def work(name, hours):
        def fn():
            calls.append(name)
            clock.t += hours * 3600
            return {"name": name}

        return fn

    kw = dict(out_dir=tmp_path, log=log, t0=0.0, cu_cap=3.0, rate=2.2, clock=clock)
    assert phasec.run_block("a", 1.0, work("a", 0.5), **kw)["name"] == "a"  # 1.1 CU used after
    assert phasec.run_block("b", 1.5, work("b", 0.5), **kw)["name"] == "b"  # 1.1 + 1.5 <= 3
    assert phasec.run_block("c", 1.0, work("c", 0.1), **kw) is None  # 2.2 + 1.0 > 3: stop
    assert phasec.run_block("d", 0.1, work("d", 0.0), **kw) is None  # skipped after the stop
    assert calls == ["a", "b"]
    assert log["blocks"]["c"]["status"] == "skipped"
    assert "cap" in log["blocks"]["c"]["reason"]
    assert log["blocks"]["d"]["reason"] == "an earlier block hit the cap"
    saved = json.loads((tmp_path / "block_a.json").read_text())
    assert saved["_block"]["minutes"] == 30.0
    log2 = {}
    again = phasec.run_block("a", 9.0, work("a", 0), out_dir=tmp_path, log=log2, t0=0.0,
                             cu_cap=0.0, clock=clock)  # fmt: skip
    assert again["name"] == "a"
    assert log2["blocks"]["a"]["status"] == "reused"
    assert calls == ["a", "b"]


def test_a_saved_block_is_reused_only_with_the_same_inputs(tmp_path):
    clock, calls = Clock(0.0), []

    def fn():
        calls.append(1)
        return {"x": len(calls)}

    def run(prov, log):
        return phasec.run_block("a", 0.1, fn, out_dir=tmp_path, log=log, t0=0.0, cu_cap=9.0,
                                clock=clock, provenance=prov)  # fmt: skip

    prov = {"git_commit": "abc", "checkpoint": "s1", "tars": {"p2": "t1"}}
    assert run(prov, {})["x"] == 1
    log = {}
    assert run(dict(prov), log)["x"] == 1
    assert log["blocks"]["a"]["status"] == "reused"
    log = {}
    assert run({**prov, "checkpoint": "s2"}, log)["x"] == 2  # another checkpoint: run again
    assert log["blocks"]["a"]["status"] == "rerun"
    assert log["blocks"]["a"]["changed_inputs"] == ["checkpoint"]
    log = {}
    assert run({**prov, "checkpoint": "s2"}, log)["x"] == 2
    assert log["blocks"]["a"]["status"] == "reused"
    assert calls == [1, 1]


def test_a_docs_only_commit_keeps_the_code_hashes_and_reuses_blocks(tmp_path):
    repo = tmp_path / "repo"
    for d in ("src", "configs", "notebooks", "docs"):
        (repo / d).mkdir(parents=True)
    paths = ("src", "configs", "notebooks/17.ipynb")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )

    def commit(path, text, msg):
        (repo / path).write_text(text)
        git("add", "-A")
        git("commit", "-q", "-m", msg)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (repo / "configs" / "base.yaml").write_text("a: 1\n")
    (repo / "notebooks" / "17.ipynb").write_text("{}\n")
    commit("src/a.py", "x = 1\n", "code")
    code1, head1 = phasec.code_tree(repo, paths), git("rev-parse", "HEAD").stdout
    commit("docs/notes.md", "docs only\n", "docs")
    assert git("rev-parse", "HEAD").stdout != head1
    assert phasec.code_tree(repo, paths) == code1
    calls = []

    def fn():
        calls.append(1)
        return {}

    out = tmp_path / "out"
    kw = dict(out_dir=out, t0=0.0, cu_cap=9.0, clock=Clock(0.0))

    def run():
        log = {}
        phasec.run_block(
            "a", 0.1, fn, log=log, provenance={"code": phasec.code_tree(repo, paths)}, **kw
        )
        return log["blocks"]["a"]["status"]

    assert run() == "done"
    assert run() == "reused"  # after the docs-only commit
    for path, text in (
        ("src/a.py", "x = 2\n"),
        ("configs/base.yaml", "a: 2\n"),
        ("notebooks/17.ipynb", "{ }\n"),
    ):
        commit(path, text, f"change {path}")
        assert run() == "rerun", path
    assert len(calls) == 4


def _png(path, arr):
    Image.fromarray(arr).save(path)
    return str(path)


def test_items_from_rows_reads_masks_by_source(tmp_path):
    img = np.zeros((40, 60, 3), np.uint8)
    m = np.zeros((40, 60), np.uint8)
    m[:10, :10] = 200
    m[30:, 50:] = 100  # below 127: not edited in BR-Gen / SID-Set
    diff = m.copy()
    diff[20:25, 20:25] = 7  # an AI base value read as 0 for DiffSeg30k
    rows = [
        {"uid": "BR-Gen/Forged/LaMa/Stuff/COCO/1_stuff.png", "source": "brgen", "label": "edited",
         "path": _png(tmp_path / "a.png", img), "mask_path": _png(tmp_path / "am.png", m)},
        {"uid": "d1", "source": "diffseg30k", "label": "edited", "mask_base": "7",
         "path": _png(tmp_path / "b.png", img), "mask_path": _png(tmp_path / "bm.png", diff)},
        {"uid": "r1", "source": "genimage", "label": "real", "path": _png(tmp_path / "c.png", img)},
    ]  # fmt: skip
    a, b, c = phasec.items_from_rows(rows)
    assert a.mask.sum() == 100
    assert a.editors is None
    assert b.mask.sum() == 100 + 100
    assert not b.mask[20:25, 20:25].any()
    assert c.mask is None
    assert c.meta["condition"] == "clean"
    (small,) = phasec.items_from_rows(rows[:1], kind="resize", level=0.5)
    assert small.image.shape == (20, 30, 3)
    assert small.mask.shape == (20, 30)
    assert small.meta["condition"] == "resize_0.5"


def _rec(head, label, pred, p_real, **kw):
    return {"head": head, "label": label, "label_pred": pred,
            "probs": {"real": p_real, "generated": 1 - p_real, "edited": 0.0}, **kw}  # fmt: skip


def test_p4_summary():
    rs = [_rec("m", "real", "real", 0.9), _rec("m", "real", "generated", 0.2),
          _rec("m", "fake", "generated", 0.1), _rec("x", "fake", "real", 0.9)]  # fmt: skip
    s = phasec.p4_summary(rs, "m")
    assert s["images"] == 3
    assert s["real_recall"] == 0.5
    assert s["fake_recall"] == 1.0
    assert s["acc"] == pytest.approx(2 / 3)


def test_p3b_summary_maps_tampered_to_edited():
    gt = np.zeros((10, 10), bool)
    gt[:5] = True
    prob = gt.astype(float)
    rs = [
        _rec("m", "real", "real", 0.9),
        _rec("m", "full_synthetic", "generated", 0.1),
        _rec("m", "tampered", "edited", 0.1, mask=mask_record(prob, gt)),
        _rec("m", "tampered", "real", 0.9, mask=mask_record(np.zeros((10, 10)), gt)),
    ]
    s = phasec.p3b_summary(rs)
    assert s["acc"] == 0.75
    assert s["confusion"]["edited"] == {"real": 1, "generated": 0, "edited": 1}
    assert s["f1"]["generated"] == 1.0
    assert s["f1"]["edited"] == pytest.approx(2 / 3)
    assert s["tampered_pixels"]["images"] == 2


def test_brgen_parts_and_share_bins():
    assert phasec.brgen_parts("BR-Gen/Forged/SDXL/Background/Places/x.png") == {
        "method": "SDXL", "region": "Background", "source": "Places"}  # fmt: skip
    assert phasec.brgen_parts("odd")["method"] == "?"
    assert phasec.edited_share_bin({"gt_pixels": 0, "pixels": 100}) == "<1%"
    assert phasec.edited_share_bin({"gt_pixels": 3, "pixels": 100}) == "1-5%"
    assert phasec.edited_share_bin({"gt_pixels": 100, "pixels": 100}) == ">40%"


def test_p3_summary_keeps_sdxl_apart():
    gt = np.zeros((10, 10), bool)
    gt[:2] = True
    good = mask_record(gt.astype(float), gt)
    bad = mask_record(np.zeros((10, 10)), gt)
    rs = [
        {"uid": "BR-Gen/Forged/LaMa/Stuff/COCO/a.png", "label_pred": "edited", "mask": good},
        {"uid": "BR-Gen/Forged/SDXL/Stuff/ImageNet/b.png", "label_pred": "real", "mask": bad},
    ]
    s = phasec.p3_summary(rs)
    assert s["all"]["images"] == 2
    assert s["unseen_methods"]["images"] == 1
    assert s["unseen_methods"]["iou_edited"] == 1.0
    assert s["sdxl"]["iou_edited"] == 0.0
    assert set(s["by_method"]) == {"LaMa", "SDXL"}
    assert s["imagenet_places_unseen"]["images"] == 0


def test_p5_summary_per_condition():
    rs = [
        _rec("det", "generated", "generated", 0.1, source="genimage", condition="clean"),
        _rec("det", "real", "real", 0.9, source="genimage", condition="clean"),
        _rec("det", "generated", "real", 0.9, source="genimage", condition="jpeg_50"),
    ]
    s = phasec.p5_summary(rs)
    assert s["det"]["clean"]["genimage"]["acc"] == 1.0
    assert s["det"]["jpeg_50"]["genimage"]["acc"] == 0.0


def test_a_failing_block_is_logged_and_the_next_block_runs(tmp_path):
    clock, log = Clock(0.0), {}

    def boom():
        clock.t += 120
        raise RuntimeError("CUDA out of memory")

    kw = dict(out_dir=tmp_path, log=log, t0=0.0, cu_cap=9.0, clock=clock)
    assert phasec.run_block("a", 0.1, boom, **kw) is None
    entry = log["blocks"]["a"]
    assert entry["status"] == "failed"
    assert entry["minutes"] == 2.0
    assert entry["error"] == "RuntimeError: CUDA out of memory"
    assert "Traceback" in entry["traceback"]
    assert not (tmp_path / "block_a.json").exists()  # nothing saved: a later session reruns it
    assert not log.get("stopped")
    stats = {
        "stats": {
            "raw": {"images": 30, "seconds": 10.0},
            "harmonized": {"images": 10, "seconds": 10},
        }
    }
    assert phasec.run_block("b", 0.1, lambda: stats, **kw)["_block"]["images_per_s"] == 2.0
    assert log["blocks"]["b"] == {"status": "done", "minutes": 0.0, "images_per_s": 2.0}


def test_the_session_cap_subtracts_what_earlier_phase_c_sessions_spent():
    assert phasec.session_cap(9.7, 23.43, 23.43) == 9.7  # first session
    assert phasec.session_cap(9.7, 23.43, 19.0) == pytest.approx(5.27)  # resumed after 4.43 CU
    assert phasec.session_cap(9.7, 23.43, 24.0) == 9.7  # a balance that went up spends nothing
    with pytest.raises(ValueError, match="cap reached"):
        phasec.session_cap(9.7, 23.43, 13.73)
    clock, log = Clock(0.0), {}
    cap = phasec.session_cap(3.0, 10.0, 8.0)  # 1.0 left this session
    kw = dict(out_dir=Path("/nonexistent"), log=log, t0=0.0, cu_cap=cap, clock=clock)
    assert phasec.run_block("x", 1.5, lambda: {}, **kw) is None
    assert log["blocks"]["x"]["status"] == "skipped"


def test_prefetch_keeps_order_and_items_and_raises_worker_errors(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(25):
        arr = (rng.random((int(rng.integers(8, 40)), 16, 3)) * 255).astype(np.uint8)
        rows.append(
            {"uid": f"u{i}", "source": "genimage", "path": _png(tmp_path / f"{i}.png", arr)}
        )
    plain = list(phasec.items_from_rows(rows, kind="noise", level=5.0))
    fetched = list(
        phasec.prefetch(phasec.item_tasks(rows, kind="noise", level=5.0), workers=6, depth=4)
    )
    assert [x.uid for x in fetched] == [x.uid for x in plain]
    assert all(np.array_equal(a.image, b.image) for a, b in zip(fetched, plain, strict=True))
    assert list(phasec.prefetch([1, lambda: 2, 3], workers=2, depth=1)) == [1, 2, 3]

    def bad():
        raise ValueError("cannot decode u3")

    out = phasec.prefetch([lambda: 0, lambda: 1, bad, lambda: 3], workers=3, depth=8)
    assert next(out) == 0
    assert next(out) == 1
    with pytest.raises(ValueError, match="cannot decode u3"):
        next(out)


def test_gpu_setup_allows_a_t4_only_after_a_passed_parity_check():
    assert phasec.gpu_setup("L4") == {"gpu": "L4", "dtype": "bf16",
                                      "probes": {"unified": True, "detection": True}}  # fmt: skip
    t4 = phasec.gpu_setup("T4", {"pass": True, "probe_pass": True})
    assert t4["dtype"] == "fp16"
    assert t4["probes"] == {"unified": True, "detection": False}  # no L4 reference for detection
    assert phasec.gpu_setup("T4", {"pass": True, "probe_pass": False})["probes"]["unified"] is False
    for parity in (None, {"pass": False}, {}):
        with pytest.raises(ValueError, match="parity"):
            phasec.gpu_setup("T4", parity)
    with pytest.raises(ValueError, match="GPU must be"):
        phasec.gpu_setup("A100")


def test_estimates_scale_by_time_and_rate():
    est = phasec.scale_estimates({"a": 2.2, "b": 1.1}, time_ratio=2.0, rate_from=2.2, rate_to=1.28)
    assert est == {"a": 2.56, "b": 1.28}
