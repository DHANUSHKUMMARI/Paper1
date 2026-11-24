"""CP6 code on the CPU: prompts, the rule-based verifier, retries, metrics and the A5 variants
(MockLLM; no GPU, no vLLM)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from eviground.explain import llm, prompts, variants, verifier


def _record(verdict="locally_edited", e_conf=0.82):
    regions = []
    if verdict == "locally_edited":
        regions = [
            {
                "id": "E1",
                "bbox": [10, 20, 30, 40],
                "area_pct": 12.34,
                "mask_conf": e_conf,
                "attr_mass": 0.61,
                "freq_anomaly": 1.8,
                "noise_anomaly": 2.1,
            }
        ]
    elif verdict == "fully_generated":
        regions = [
            {
                "id": "A1",
                "bbox": [0, 0, 50, 50],
                "area_pct": 9.0,
                "mask_conf": 0.02,
                "attr_mass": 0.44,
                "freq_anomaly": 1.1,
                "noise_anomaly": 1.0,
            }
        ]
    return {
        "prediction": {"real": 0.03, "fully_generated": 0.05, "locally_edited": 0.92}
        if verdict == "locally_edited"
        else {"real": 0.1, "fully_generated": 0.85, "locally_edited": 0.05},
        "verdict": verdict,
        "global": [
            {"id": "G1", "type": "branch_reliance", "semantic": 0.7, "trace": 0.3},
            {"id": "G2", "type": "image_freq_anomaly", "value": 0.123},
            {"id": "G3", "type": "image_noise_anomaly", "value": 4.5},
        ],
        "regions": regions,
    }


def _claim(ctype, text, ids):
    return {"claim_type": ctype, "text": text, "evidence_ids": ids}


def _rules(claim, rec):
    return {r for r, _ in verifier.check_claim(claim, rec)}


# ---- prompts -------------------------------------------------------------------------------


def test_grounded_input_holds_no_pixels_paths_or_uid(tmp_path):
    from eviground.explain import evidence

    rng = np.random.default_rng(0)
    image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    mask = np.zeros((64, 64))
    mask[10:30, 10:30] = 0.9
    probs = {"real": 0.05, "generated": 0.05, "edited": 0.9}
    rec, rles = evidence.evidence_record(probs, 0.6, mask, mask, image)
    uid = "sidset/validation/tampered/images/tampered_00140.png"
    evidence.save_bundle(tmp_path, uid, rec, rles, {"decoder": mask}, probs, 0.6)
    bundle = evidence.load_bundle(tmp_path, uid)
    text = prompts.grounded_prompt({**bundle, **bundle["evidence"]})
    for bad in (
        str(tmp_path),
        uid,
        "tampered_",
        ".png",
        ".npz",
        "data:",
        "base64",
        "/content",
        "rle",
        "counts",
    ):
        assert bad not in text, bad
    assert not __import__("re").search(r"(\d+(\.\d+)?,){20,}", text), "array-like run of numbers"
    ev = llm.evidence_from_prompt(text)
    assert set(ev) <= set(prompts.EVIDENCE_KEYS)


def test_parse_report_refuses_non_schema_answers():
    with pytest.raises(prompts.ReportError):
        prompts.parse_report("no json here")
    with pytest.raises(prompts.ReportError):
        prompts.parse_report('{"verdict": "maybe", "claims": []}')
    rep = prompts.parse_report('Sure: {"verdict": "real", "claims": [{"text": "x"}]} done')
    assert rep["claims"][0]["evidence_ids"] == []


# ---- verifier rules -----------------------------------------------------------------------


def test_rule_a_missing_and_unknown_ids():
    rec = _record()
    assert "a" in _rules(_claim("global", "Noise is uneven.", []), rec)
    assert "a" in _rules(_claim("global", "Noise is uneven.", ["G9"]), rec)
    assert not _rules(_claim("global", "The frequency anomaly is 0.123.", ["G2"]), rec)


def test_rule_b_rounding_aware_tolerance():
    rec = _record()
    ok = [
        "The semantic weight is 0.7.",
        "The semantic weight is 0.71.",
        "It holds 92% probability.",
        "E1 covers 12.3% of the image.",
        "E1 covers 13% of the image.",
    ]
    for text in ok:
        ids = ["E1", "PRED", "G1"]
        assert "b" not in _rules(_claim("edited_region", text, ids), rec), text
    bad = ["The semantic weight is 0.75.", "E1 covers 14% of the image.", "Probability 80%."]
    for text in bad:
        assert "b" in _rules(_claim("edited_region", text, ["E1", "PRED", "G1"]), rec), text


def test_rule_b_ignores_evidence_ids_in_the_text():
    rec = _record()
    assert not _rules(_claim("edited_region", "Region E1 is marked as edited.", ["E1"]), rec)


def test_rule_c_tau_for_e_regions():
    weak, strong = _record(e_conf=0.55), _record(e_conf=0.65)
    c = _claim("edited_region", "E1 is edited.", ["E1"])
    assert "c" in _rules(c, weak)
    assert "c" not in _rules(c, strong)


def test_rule_c_prime_for_a_regions():
    rec = _record("fully_generated")
    assert not _rules(
        _claim("attribution", "The model's evidence concentrates in A1.", ["A1"]), rec
    )
    assert "c_prime" in _rules(_claim("edited_region", "A1 is suspicious.", ["A1"]), rec)
    assert "c_prime" in _rules(_claim("attribution", "A1 was edited.", ["A1"]), rec)


def test_schema_claim_type():
    assert "schema" in _rules(
        _claim("opinion", "The frequency anomaly is 0.123.", ["G2"]), _record()
    )


# ---- runs, retries, metrics --------------------------------------------------------------


def test_clean_mock_reports_pass_first_time():
    recs = [("u1", _record()), ("u2", _record("fully_generated"))]
    res = verifier.run_grounded(llm.MockLLM(), recs)
    assert all(r["retries"] == 0 and r["dropped_claims"] == 0 for r in res)
    m = verifier.first_pass_metrics(res)
    assert m["gr"] == 1.0
    assert m["hr_e"] == 0.0


@pytest.mark.parametrize(
    "fault,rule", [("bad_id", "a"), ("bad_number", "b"), ("a_edited", "c_prime")]
)
def test_each_fault_is_caught_and_repaired_on_retry(fault, rule):
    rec = _record("fully_generated") if fault == "a_edited" else _record()
    mock = llm.MockLLM({0: fault})
    (res,) = verifier.run_grounded(mock, [("u", rec)])
    assert rule in {e[0] for c in res["first_checks"]["failed"] for e in c["errors"]}
    assert res["retries"] == 1
    assert res["dropped_claims"] == 0
    assert len(mock.calls) == 2
    m = verifier.first_pass_metrics([res])
    assert m["gr"] < 1.0
    assert m["rejections_by_rule"][rule] == 1


def test_weak_region_claim_counted_under_rule_c_for_e():
    (res,) = verifier.run_grounded(llm.MockLLM({0: "weak_region"}), [("u", _record(e_conf=0.55))])
    m = verifier.first_pass_metrics([res])
    assert m["rejections_by_rule_and_region"]["c:E"] >= 1


def test_retries_stop_at_two_and_failing_claims_are_dropped():
    mock = llm.MockLLM({0: "bad_id"}, repair=False)
    (res,) = verifier.run_grounded(mock, [("u", _record())])
    assert res["retries"] == verifier.MAX_RETRIES
    assert len(mock.calls) == 1 + verifier.MAX_RETRIES
    assert res["dropped_claims"] == 1
    assert all(not verifier.check_claim(c, _record()) for c in res["final_report"]["claims"])
    assert res["first_report"]["claims"] != res["final_report"]["claims"]  # both stored


def test_unparseable_first_answer_is_retried_and_counted():
    (res,) = verifier.run_grounded(llm.MockLLM({0: "bad_json"}), [("u", _record())])
    assert res["first_report"] is None
    assert res["final_report"] is not None
    assert verifier.first_pass_metrics([res])["unparsed_first_answers"] == 1


def test_batched_first_pass_is_one_call():
    mock = llm.MockLLM()
    verifier.run_grounded(mock, [(f"u{i}", _record()) for i in range(5)])
    assert len(mock.calls) == 1
    assert len(mock.calls[0]) == 5


# ---- HR-GT and variants ------------------------------------------------------------------


def test_hr_gt_verdict_and_region_boxes():
    mask = np.zeros((100, 100), bool)
    mask[20:60, 10:40] = True  # E1's box [10, 20, 30, 40] lies inside
    rec = _record()
    rep = variants.template_report(rec)
    assert verifier.hr_gt_counts(rep, "edited", mask, rec) == {"checkable": 2, "wrong": 0}
    assert verifier.hr_gt_counts(rep, "real", None, rec) == {"checkable": 2, "wrong": 2}
    img_rep = {
        "verdict": "locally_edited",
        "claims": [
            {
                "claim_type": "edited_region",
                "text": "x",
                "evidence_ids": [],
                "box": [70, 70, 20, 20],
            }
        ],
    }
    assert verifier.hr_gt_counts(img_rep, "edited", mask) == {"checkable": 2, "wrong": 1}
    assert verifier.hr_gt_counts(None, "edited", mask) == {"checkable": 1, "wrong": 1}


def test_template_report_passes_the_verifier():
    for rec in (_record(), _record("fully_generated")):
        rep = variants.template_report(rec)
        assert all(not verifier.check_claim(c, rec) for c in rep["claims"]), rep


def test_image_baseline_uses_the_same_schema():
    class FakeVL:
        def generate_with_images(self, prompts_, images):
            assert all("box" in p for p in prompts_)
            return [json.dumps({"verdict": "real", "claims": [], "limitations": ""}), "garbage"]

    out = variants.image_reports(FakeVL(), [object(), object()])
    assert out[0]["report"]["verdict"] == "real"
    assert out[1]["report"] is None


def test_the_cp6_modules_never_load_the_detector():
    code = (
        "import sys; from eviground.explain import llm, prompts, verifier, variants, evidence; "
        "from eviground.eval import p6cache; "
        "assert not any(m.startswith('eviground.models') for m in sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pinned_revisions_are_commit_shas():
    for repo, rev in (llm.QWEN25_AWQ, llm.QWEN2VL_AWQ):
        assert repo.endswith("-AWQ")
        assert len(rev) == 40
        int(rev, 16)


def test_hr_gt_summary_counts_images_and_verdicts():
    mask = np.zeros((100, 100), bool)
    mask[20:60, 10:40] = True
    rec = _record()
    rep = variants.template_report(rec)
    s = verifier.hr_gt_summary([(rep, "edited", mask, rec), (None, "real", None, None)])
    assert s["images"] == 2
    assert s["checkable_claims"] == 3
    assert s["wrong_claims"] == 1
    assert s["verdict_accuracy"] == 0.5


def test_a_verdict_that_overrules_the_evidence_is_retried_then_replaced():
    rec = _record()
    (res,) = verifier.run_grounded(llm.MockLLM({0: "bad_verdict"}), [("u", rec)])
    assert res["first_checks"]["verdict_error"]
    assert res["retries"] == 1
    assert res["final_report"]["verdict"] == "locally_edited"
    assert not res["verdict_replaced"]
    assert verifier.first_pass_metrics([res])["verdict_mismatch_first_answers"] == 1
    (stuck,) = verifier.run_grounded(llm.MockLLM({0: "bad_verdict"}, repair=False), [("u", rec)])
    assert stuck["verdict_replaced"]
    assert stuck["final_report"]["verdict"] == "locally_edited"
    assert stuck["final_report"]["confidence"] == rec["prediction"]["locally_edited"]


def test_attribution_claims_are_not_scored_against_an_edit_mask():
    rec = _record("fully_generated")
    rep = variants.template_report(rec)
    assert verifier.hr_gt_counts(rep, "generated", None, rec) == {"checkable": 1, "wrong": 0}


def test_off_image_box_area_counts_as_outside():
    mask = np.ones((100, 100), bool)
    assert verifier.box_inside_fraction([90, 0, 100, 10], mask) == pytest.approx(0.1)
    assert verifier.box_inside_fraction([0, 0, 10, 10], mask) == 1.0
