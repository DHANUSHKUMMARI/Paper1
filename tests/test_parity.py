"""T4/L4 parity check (Dhanush, 25 Nov): the frozen sample and the three criteria."""

import pytest

from eviground.eval import parity


def test_sample_is_seeded_sorted_and_hashed():
    edited = [f"e{i}" for i in range(500)]
    real = [f"r{i}" for i in range(40)]
    a = parity.sample_parity(edited, real, n=300)
    b = parity.sample_parity(list(reversed(edited)), real, n=300)
    assert a == b
    assert a["edited"]["images"] == 300
    assert a["real"]["images"] == 40  # fewer than n: all of them
    assert a["edited"]["uids"] == sorted(a["edited"]["uids"])
    assert parity.intact(a)
    a["edited"]["uids"][0] = "changed"
    assert not parity.intact(a)
    assert parity.sample_parity(edited, real, n=300, salt="other")["sha256"] != b["sha256"]


def _refs():
    e = {f"e{i}": 0.5 + i / 100 for i in range(10)}
    r = {f"r{i}": "real" for i in range(200)}
    return e, r


def test_identical_results_pass():
    e, r = _refs()
    v = parity.parity_verdict(parity.L4_P2_IOU, dict(e), dict(r), e, r)
    assert v["pass"]
    assert v["edited_per_image_iou"]["median_abs_diff"] == 0.0
    assert v["real_verdict_agreement"]["agreement"] == 1.0


def test_each_criterion_can_fail_on_its_own():
    e, r = _refs()
    v = parity.parity_verdict(parity.L4_P2_IOU + 0.006, dict(e), dict(r), e, r)
    assert not v["p2_pooled_iou"]["pass"]
    assert not v["pass"]
    r2 = dict(r, r0="edited", r1="edited")  # 198/200 = 0.99 < 0.995
    v = parity.parity_verdict(parity.L4_P2_IOU, dict(e), r2, e, r)
    assert not v["real_verdict_agreement"]["pass"]
    assert v["real_verdict_agreement"]["disagreeing_uids"] == ["r0", "r1"]
    e2 = {u: x + 0.02 for u, x in e.items()}
    v = parity.parity_verdict(parity.L4_P2_IOU, e2, dict(r), e, r)
    assert not v["edited_per_image_iou"]["pass"]
    assert v["p2_pooled_iou"]["pass"]


def test_a_missing_t4_result_is_an_error():
    e, r = _refs()
    with pytest.raises(ValueError, match="no T4 result"):
        parity.parity_verdict(parity.L4_P2_IOU, {}, dict(r), e, r)


def test_the_autocast_dtype_can_be_set_and_is_bf16_by_default():
    import torch

    from eviground.eval import runner

    assert runner.AMP_DTYPE == "bf16"
    try:
        assert runner.set_amp_dtype("fp16") == "fp16"
        with pytest.raises(ValueError, match="unknown autocast dtype"):
            runner.set_amp_dtype("fp8")
        assert runner.AMP_DTYPE == "fp16"
    finally:
        runner.set_amp_dtype("bf16")
    with runner.autocast(torch.device("cpu")):  # nothing on the CPU
        assert torch.ones(1).dtype == torch.float32
