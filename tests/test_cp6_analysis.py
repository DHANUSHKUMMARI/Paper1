"""CP6 analysis after Phase D (Dhanush's review, 25 Nov): the template fix, the HR-GT split, the
all-box sensitivity, limitation rule counts, criterion 1, the confusion matrix, box counts."""

import numpy as np
import pytest

from eviground.explain import cp6analysis, evidence, variants, verifier


def _real_shaped(verdict, rng, confs=(0.55, 0.9)):
    """An evidence record built by the real M5 code (evidence.evidence_record), with one E
    region per value in ``confs`` (mask probability inside the blob) for an edited verdict."""
    h, w = 96, 128
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    mask_prob = np.zeros((h, w))
    for i, conf in enumerate(confs):
        mask_prob[10 + 40 * i : 40 + 40 * i, 10:50] = conf
    attr = rng.random((h, w))
    probs = {
        "edited": {"real": 0.03, "generated": 0.05, "edited": 0.92},
        "generated": {"real": 0.1, "generated": 0.85, "edited": 0.05},
        "real": {"real": 0.8, "generated": 0.15, "edited": 0.05},
    }[verdict]
    rec, _ = evidence.evidence_record(probs, float(rng.random()), mask_prob, attr, img)
    return rec


def _failing(rep, rec):
    return [
        (c["text"], verifier.check_claim(c, rec))
        for c in rep["claims"]
        if verifier.check_claim(c, rec)
    ]


def test_the_fixed_template_passes_the_verifier_on_real_shaped_records():
    rng = np.random.default_rng(0)
    for _ in range(30):
        for verdict in ("edited", "generated", "real"):
            confs = tuple(rng.uniform(0.5, 1.0, size=rng.integers(0, 3)))
            rec = _real_shaped(verdict, rng, confs)
            rep = variants.template_report(rec)
            assert not _failing(rep, rec), (rec, _failing(rep, rec))


def test_the_as_run_template_broke_rule_c_on_low_confidence_regions():
    rec = _real_shaped("edited", np.random.default_rng(1), confs=(0.55, 0.9))
    assert [r["mask_conf"] for r in rec["regions"]] == [0.55, 0.9]
    as_run = variants.template_report(rec, min_mask_conf=None)
    assert [e[0] for _, errs in _failing(as_run, rec) for e in errs] == ["c"]
    fixed = variants.template_report(rec)
    assert [c["evidence_ids"] for c in fixed["claims"] if c["claim_type"] == "edited_region"] == [
        ["E2"]
    ]
    counts = cp6analysis.rule_counts([(as_run, rec), (fixed, rec)])
    assert counts["reports_passing"] == 1
    assert counts["failing_claims_by_rule"] == {"c": 1}
    assert counts["failing_claims_by_type"] == {"edited_region": 1}


def _mask():
    m = np.zeros((100, 100), bool)
    m[:50, :50] = True
    return m


def _rec_with_regions():
    return {
        "prediction": {"real": 0.1, "fully_generated": 0.1, "locally_edited": 0.8},
        "verdict": "locally_edited",
        "global": [{"id": "G1", "semantic": 0.5, "trace": 0.5}],
        "regions": [
            {"id": "E1", "bbox": [0, 0, 40, 40], "mask_conf": 0.9, "area_pct": 16.0},
            {"id": "E2", "bbox": [60, 60, 30, 30], "mask_conf": 0.9, "area_pct": 9.0},
        ],
    }


def test_hr_gt_is_split_into_verdict_and_region_claims():
    rec = _rec_with_regions()
    region = [
        {"claim_type": "edited_region", "text": "", "evidence_ids": [i]} for i in ("E1", "E2")
    ]
    right = {"verdict": "locally_edited", "claims": region}
    wrong = {"verdict": "real", "claims": region}
    c = verifier.hr_gt_counts(wrong, "edited", _mask(), rec)
    assert (c["verdict_wrong"], c["region_checkable"], c["region_wrong"]) == (1, 2, 1)
    assert c["checkable"] == 3 and c["wrong"] == 2
    s = verifier.hr_gt_summary([(right, "edited", _mask(), rec), (wrong, "edited", _mask(), rec),
                                (None, "real", None, rec)])  # fmt: skip
    assert s["verdict_claims"] == {"checkable": 3, "wrong": 2, "rate": pytest.approx(2 / 3)}
    assert s["region_claims"] == {"checkable": 4, "wrong": 2, "rate": 0.5}
    assert s["checkable_claims"] == 3 + 4 and s["wrong_claims"] == 2 + 2


def test_score_all_boxes_scores_boxed_claims_of_any_type_as_a_sensitivity_check():
    rep = {"verdict": "locally_edited", "claims": [
        {"claim_type": "edited_region", "text": "", "box": [0, 0, 40, 40]},
        {"claim_type": "global", "text": "", "box": [60, 60, 30, 30]},
        {"claim_type": "global", "text": "no box"},
    ]}  # fmt: skip
    primary = verifier.hr_gt_counts(rep, "edited", _mask())
    assert (primary["region_checkable"], primary["region_wrong"]) == (1, 0)
    every = verifier.hr_gt_counts(rep, "edited", _mask(), score_all_boxes=True)
    assert (every["region_checkable"], every["region_wrong"]) == (2, 1)
    # EviGround claims carry no box of their own, so the option does not change them
    rec = _rec_with_regions()
    ev = {
        "verdict": "locally_edited",
        "claims": [{"claim_type": "attribution", "evidence_ids": ["E1"]}],
    }
    assert (
        verifier.hr_gt_counts(ev, "edited", _mask(), rec, score_all_boxes=True)["region_checkable"]
        == 0
    )


def test_limitation_failures_are_counted_by_rule():
    rec = _rec_with_regions()
    checked = [
        {"claim": {"claim_type": "limitation", "text": "about 7 things"}, "errors": [("b", "x")]},
        {"claim": {"claim_type": "limitation", "evidence_ids": ["E1"]},
         "errors": [("limitation", "x"), ("b", "y")]},
        {"claim": {"claim_type": "limitation"}, "errors": []},
    ]  # fmt: skip
    res = {"first_checks": {"checked": checked, "verdict_error": None, "confidence_error": None}}
    m = verifier.first_pass_metrics([res])
    assert m["limitation_claims"] == {"failed": 2, "passed": 1}
    assert m["limitation_rejections_by_rule"] == {"b": 2, "limitation": 1}
    assert m["claims"] == 0  # left out of GR / HR-E
    assert rec  # the record is not needed for the tally


def test_final_check_passes_only_when_every_final_claim_passes():
    rec = _rec_with_regions()
    claim = {"claim_type": "edited_region", "text": "E1 is edited.", "evidence_ids": ["E1"]}
    good = {"verdict": "locally_edited", "confidence": 0.8, "claims": [claim]}
    ok = cp6analysis.final_check([{"uid": "a", "final_report": good}], {"a": rec})
    assert ok["passes"] and ok["claims"] == 1 and ok["failing_claims"] == 0
    bad = {**good, "claims": [{"claim_type": "edited_region", "text": "", "evidence_ids": ["E9"]}]}
    out = cp6analysis.final_check(
        [{"uid": "a", "final_report": bad}, {"uid": "b", "final_report": None}],
        {"a": rec, "b": rec},
    )
    assert not out["passes"]
    assert out["failing_claims_by_rule"] == {"a": 1}
    assert out["reports_without_a_final_report"] == 1


def test_verdict_confusion_and_boxes_by_claim_type():
    rows = [({"verdict": "real"}, "real"), ({"verdict": "real"}, "edited"),
            ({"verdict": "odd"}, "generated"), (None, "generated")]  # fmt: skip
    conf = cp6analysis.verdict_confusion(rows)
    assert conf["real"]["real"] == 1
    assert conf["locally_edited"]["real"] == 1
    assert conf["fully_generated"]["unparsed"] == 2
    assert sum(sum(r.values()) for r in conf.values()) == 4
    reps = [{"claims": [{"claim_type": "global", "box": [0, 0, 1, 1]},
                        {"claim_type": "edited_region", "box": [0, 0, 1, 1]},
                        {"claim_type": "verdict"}]}, None]  # fmt: skip
    assert cp6analysis.boxes_by_claim_type(reps) == {
        "edited_region": {"with_box": 1, "without_box": 0},
        "global": {"with_box": 1, "without_box": 0},
        "verdict": {"with_box": 0, "without_box": 1},
    }


def test_read_entry_carries_each_claims_errors():
    rec = _rec_with_regions()
    first = {"verdict": "locally_edited", "claims": [
        {"claim_type": "edited_region", "text": "", "evidence_ids": ["E9"]}]}  # fmt: skip
    final = {"verdict": "locally_edited", "confidence": 0.8, "claims": []}
    e = cp6analysis.read_entry("u", rec, {"first_report": first, "final_report": final,
                                          "retries": 1}, "edited")  # fmt: skip
    assert e["true_verdict"] == "locally_edited"
    assert e["first_pass"]["claims"][0]["errors"][0][0] == "a"
    assert e["final"]["claims"] == []
    assert all(not c["errors"] for c in e["template"]["claims"])
    assert e["image_baseline"] is None
