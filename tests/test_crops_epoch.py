"""Training crop rule and epoch draws (decision of 14 Nov 2026)."""

from __future__ import annotations

import numpy as np
import pytest

from eviground.data import crops, epoch, sampler

SIZE = 256


def _blob(h=1024, w=1024, top=508, left=508, bh=8, bw=8):
    m = np.zeros((h, w), dtype=np.uint8)
    m[top : top + bh, left : left + bw] = 3
    return m


def _share(mask, box):
    left, top, right, bottom = box
    return float((mask[top:bottom, left:right] > 0).mean())


def test_edited_pixel_position_in_crop_is_uniform_not_centred():
    mask = _blob()
    rng = np.random.default_rng(0)
    rel_x, rel_y = [], []
    for _ in range(4000):
        box, _, k = crops.edited_crop(mask, rng, SIZE, share=(0.0, 1.0))
        assert box is not None
        assert k == 1
        left, top, right, bottom = box
        assert 0 <= left < right <= 1024
        assert 0 <= top < bottom <= 1024
        rel_x.append(512 - left)  # blob centre inside the crop
        rel_y.append(512 - top)
    rel_x, rel_y = np.array(rel_x), np.array(rel_y)
    for rel in (rel_x, rel_y):
        assert abs(rel.mean() - 128) < 6  # uniform over the crop: mean near the middle ...
        assert 66 < rel.std() < 82  # ... but spread like a uniform (sd 73.9), not peaked
        counts = np.histogram(rel, bins=4, range=(0, 256))[0]
        assert counts.min() > 850  # each quarter of the crop ~1,000 of 4,000
    central = np.mean((np.abs(rel_x - 128) < 32) & (np.abs(rel_y - 128) < 32))
    assert central < 0.10  # uniform: 6.25% in the central 64 x 64; a centred crop: 100%


def test_window_stays_inside_the_image_near_a_border():
    mask = _blob(h=300, w=400, top=0, left=390, bh=10, bw=10)
    rng = np.random.default_rng(1)
    for _ in range(500):
        box, _, _ = crops.edited_crop(mask, rng, SIZE, share=(0.0, 1.0))
        left, top, right, bottom = box
        assert 0 <= left < right <= 400
        assert 0 <= top < bottom <= 300
        assert _share(mask, box) > 0  # the drawn edited pixel is inside


def test_share_check_holds_on_every_returned_box():
    rng = np.random.default_rng(2)
    for s in range(30):
        m = np.zeros((600, 700), dtype=np.uint8)
        r = np.random.default_rng(s)
        for _ in range(int(r.integers(1, 5))):
            t, left = int(r.integers(0, 500)), int(r.integers(0, 600))
            m[t : t + int(r.integers(5, 300)), left : left + int(r.integers(5, 300))] = 1
        box, share, _ = crops.edited_crop(m, rng, SIZE)
        if box is not None:
            assert 0.01 <= _share(m, box) <= 0.90
            assert share == pytest.approx(_share(m, box))


def test_min_pixels_keeps_small_edits():
    mask = _blob(bh=15, bw=20)  # 300 edited pixels: 0.46% of a crop, below the 1% rule
    rng = np.random.default_rng(3)
    box, _, tries = crops.edited_crop(mask, rng, SIZE)
    assert box is None
    assert tries == crops.TRIES
    box, share, _ = crops.edited_crop(mask, rng, SIZE, min_pixels=256)
    assert box is not None
    assert share * SIZE * SIZE >= 256
    assert not crops.has_valid_window(mask, SIZE)
    assert crops.has_valid_window(mask, SIZE, min_pixels=256)


def test_empty_or_small_masks_fail_cleanly():
    rng = np.random.default_rng(4)
    assert crops.edited_crop(np.zeros((300, 300), np.uint8), rng) == (None, 0.0, 0)
    assert crops.edited_crop(np.ones((200, 300), np.uint8), rng) == (None, None, 0)
    assert not crops.has_valid_window(np.ones((200, 300), np.uint8))
    assert not crops.has_valid_window(np.ones((300, 300), np.uint8))  # 100% everywhere


def test_has_valid_window_matches_brute_force():
    size, share = 8, (0.1, 0.6)
    for s in range(40):
        r = np.random.default_rng(s)
        m = (r.random((14, 17)) < r.uniform(0.02, 0.9)).astype(np.uint8)
        brute = any(
            share[0] <= m[t : t + size, x : x + size].mean() <= share[1]
            for t in range(14 - size + 1)
            for x in range(17 - size + 1)
        )
        assert crops.has_valid_window(m, size, share) == brute
        assert crops.window_counts(m, size).shape == (7, 10)


def _pools():
    """Sources and labels covering every unified pool; edited images e0-e9, half never crop."""
    rows = [("genimage", "real")] * 6 + [("sidset", "real")] * 4 + [("coco", "real")] * 5
    rows += [("diffseg30k", "real")] * 3
    rows += [("genimage", "generated")] * 6 + [("sidset", "generated")] * 4
    rows += [("diffseg30k", "edited")] * 10
    sources = [s for s, _ in rows]
    labels = [lab for _, lab in rows]
    uids = [f"r{i}" for i in range(len(rows) - 10)] + [f"e{i}" for i in range(10)]
    return sources, labels, uids


def test_replacement_keeps_the_class_and_source_balance():
    sources, labels, uids = _pools()
    bad = {f"e{i}" for i in range(5)}  # these edited images never give a valid crop

    def crop_fn(i, rng):
        return None if uids[i] in bad else (0, 0, SIZE, SIZE)

    plan = epoch.epoch_plan(sources, labels, uids, 60, seed=42, crop_fn=crop_fn)
    expected = {f"{c}/{s}": n for (c, s), n in sampler.quotas(60).items()}
    assert plan.per_pool(sources, labels) == dict(sorted(expected.items()))
    assert all(uids[i] not in bad for i, _ in plan.items)
    assert set(plan.failures) <= bad
    assert sum(plan.failures.values()) > 0
    assert set(plan.replacements) == {"edited/diffseg30k"}
    assert plan.log()["replaced"] == sum(plan.failures.values())
    again = epoch.epoch_plan(sources, labels, uids, 60, seed=42, crop_fn=crop_fn)
    assert again.items == plan.items  # deterministic per seed


def test_replacement_limit_raises_when_no_image_can_crop():
    sources, labels, uids = _pools()

    def crop_fn(i, rng):
        return None if labels[i] == "edited" else (0, 0, SIZE, SIZE)

    with pytest.raises(RuntimeError, match="crop replacements"):
        epoch.epoch_plan(sources, labels, uids, 30, seed=1, crop_fn=crop_fn)


def test_screen_never_valid_logs_images_without_a_valid_window():
    masks = {
        "ok": _blob(bh=60, bw=60),
        "tiny": _blob(bh=4, bw=4),
        "full": np.ones((300, 300), np.uint8),
    }
    assert epoch.screen_never_valid(masks, masks.__getitem__) == ["full", "tiny"]
    assert epoch.screen_never_valid(["tiny"], masks.__getitem__, min_pixels=16) == []


def test_trace_edited_box_repeats_the_cp1_rule_exactly():
    """The diagnostic's tracing twin uses the same random numbers as crops.edited_box."""
    from eviground.data import crop_diagnostic

    fails = 0
    for s in range(60):
        r = np.random.default_rng(100 + s)
        m = np.zeros((int(r.integers(256, 700)), int(r.integers(256, 700))), np.uint8)
        t, left = int(r.integers(0, m.shape[0])), int(r.integers(0, m.shape[1]))
        m[t : t + int(r.integers(1, 400)), left : left + int(r.integers(1, 400))] = 5
        box, last, tries = crops.edited_box(m, np.random.default_rng(s))
        tr = crop_diagnostic.trace_edited_box(m, np.random.default_rng(s))
        assert tr["ok"] == (box is not None)
        assert len(tr["shares"]) == tries
        assert tr["shares"][-1] == pytest.approx(last)
        fails += not tr["ok"]
    assert 0 < fails < 60  # the cases cover both outcomes


def test_crop_diagnostic_sorts_failed_draws_by_kind(tmp_path):
    """Masks that fail below 1% (a tiny edit) and above 90% (a full edit) are counted apart."""
    import csv
    import gzip

    from PIL import Image

    from eviground.data import crop_diagnostic

    masks = {"tiny": _blob(h=512, w=512, top=250, left=250, bh=5, bw=5)}
    masks["full"] = np.full((512, 512), 4, np.uint8)
    masks["ok"] = _blob(h=512, w=512, top=100, left=100, bh=200, bw=200)
    rows = []
    for name, m in masks.items():
        p = tmp_path / f"{name}.mask.png"
        Image.fromarray(m).save(p)
        rows.append({"uid": name, "source": "diffseg30k", "label": "edited", "mask_path": p})
    for label, source in sampler.quotas(10):
        if label != "edited":
            rows.append({"uid": f"{source}-{label}", "source": source, "label": label})
    manifest = tmp_path / "manifest.csv.gz"
    with gzip.open(manifest, "wt", newline="") as fh:
        w = csv.DictWriter(fh, ["uid", "source", "label", "split", "h_action", "mask_path"])
        w.writeheader()
        w.writerows({**r, "split": "train", "h_action": "compress"} for r in rows)
    diag = crop_diagnostic.run_diagnostic(manifest, crops_per_class=40, seed=7)
    d, im = diag["draws"], diag["images"]
    kinds = d["failed_by_kind"]
    assert set(kinds) == {"all_tries_below_1pct", "all_tries_above_90pct"}
    assert sum(kinds.values()) == d["failed"]
    assert d["edited_draws"] == 40
    hist = d["failed_tries_share_histogram"]
    assert sum(hist.values()) == d["failed"] * crops.TRIES  # every failed draw used all 10 tries
    assert hist["100%"] == kinds["all_tries_above_90pct"] * crops.TRIES
    assert hist["[1%, 90%]"] == 0
    assert d["failed_below_1pct_best_try_edited_pixels"]["max"] <= 25
    assert im["edited_training_images"] == 3
    assert im["never_valid_1_90pct"] == 2  # tiny and full
    assert im["never_valid_min_256px_le_90pct"] == 2  # 25 px is below 256 too
    assert d["reproduced"] is None
