"""M6 rule-based verifier and the hallucination metrics (spec sections 8 and 10; thresholds
pre-registered on 24 Nov 2026, before CP6).

Rules on each claim:

- **schema**: ``claim_type`` is one of :data:`prompts.CLAIM_TYPES`;
- **(a)**: at least one evidence ID, and every cited ID exists in the record (``PRED``, G*, E*, A*);
- **(b)**: every number in the text matches a value of a cited entry, within
  ``max(0.01, 2% of |value|)``; a percentage is compared with value x 100 for values in 0-1 and
  with ``area_pct`` directly, within 1 percentage point;
- **(c)**: a claim citing an E region needs its ``mask_conf`` >= 0.6;
- **(c')**: a claim citing an A region must be of type ``attribution`` and must not call the
  region edited or manipulated (the decoder is about 0 on fully generated images, so mask_conf
  is not used for A regions).

Failing claims go back to the LLM with the errors, at most twice; claims still failing are then
dropped. The **first pass** and the **final** report are both kept, and GR / HR-E are measured on
the first pass (before the verifier removes anything). HR-GT compares checkable claims with the
ground truth: the verdict, and region claims (wrong when < 10% of the box is inside the mask).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from eviground.explain import prompts
from eviground.explain.llm import LLM

TAU_E = 0.6
ABS_TOL = 0.01
REL_TOL = 0.02
PCT_TOL = 1.0
MAX_RETRIES = 2
BOX_INSIDE_MIN = 0.10
EDIT_WORDS = re.compile(
    r"\b(edit(ed|s|ing)?|manipulat\w*|tamper\w*|inpaint\w*|forg(ed|ery)|altered|spliced)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)(\s*%)?")
TRUE_VERDICT = {"real": "real", "generated": "fully_generated", "edited": "locally_edited"}


def entries(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Evidence ID -> entry (``PRED`` is the prediction block)."""
    out: dict[str, Mapping[str, Any]] = {prompts.PRED_ID: record.get("prediction", {})}
    for e in [*record.get("global", []), *record.get("regions", [])]:
        out[e["id"]] = e
    return out


def _values(entry: Mapping[str, Any]) -> list[tuple[str, float]]:
    vals = []
    for k, v in entry.items():
        if isinstance(v, bool) or k == "id":
            continue
        if isinstance(v, int | float):
            vals.append((k, float(v)))
        elif isinstance(v, list | tuple):
            vals += [
                (k, float(x)) for x in v if isinstance(x, int | float) and not isinstance(x, bool)
            ]
    return vals


def number_matches(quoted: float, percent: bool, values: Iterable[tuple[str, float]]) -> bool:
    for key, v in values:
        if percent:
            if key == "area_pct" and abs(quoted - v) <= PCT_TOL:
                return True
            if 0.0 <= v <= 1.0 and abs(quoted - 100.0 * v) <= PCT_TOL:
                return True
        elif abs(quoted - v) <= max(ABS_TOL, REL_TOL * abs(v)):
            return True
    return False


def _region_ids(ids: Sequence[str]) -> list[str]:
    return [i for i in ids if i[:1] in ("E", "A")]


def _count_matches(q: float, ids: Sequence[str], record: Mapping[str, Any]) -> bool:
    """An integer equal to the number of regions cited, or of regions in the record, is a
    count ("2 edited regions") and matches (spec card, 24 Nov)."""
    return float(q).is_integer() and int(q) in {
        len(_region_ids(ids)),
        len(record.get("regions", [])),
    }


def _check_numbers(text: str, ids: Sequence[str], cited, record) -> list[tuple[str, str]]:
    scrubbed = re.sub(r"\b[PGEA]\w*\d+\b", " ", text)  # IDs like E1, G2, PRED are not numbers
    values = [kv for e in cited for kv in _values(e)]
    errors = []
    for m in NUMBER.finditer(scrubbed):
        q, pct = float(m.group(1)), bool(m.group(2))
        if number_matches(q, pct, values) or (not pct and _count_matches(q, ids, record)):
            continue
        errors.append(("b", f"the number {m.group(0).strip()} matches no value of {list(ids)}"))
    return errors


def _check_regions(ctype, text: str, ids: Sequence[str], known) -> list[tuple[str, str]]:
    errors = []
    for i in ids:
        if i not in known:
            continue
        if i.startswith("E") and float(known[i].get("mask_conf", 0.0)) < TAU_E:
            errors.append(("c", f"{i} has mask_conf {known[i].get('mask_conf')} < {TAU_E}"))
        if i.startswith("A") and (ctype != "attribution" or EDIT_WORDS.search(text)):
            errors.append(
                (
                    "c_prime",
                    f"{i} is an attribution region; it may only be described as "
                    "where the model's evidence concentrates, never as edited",
                )
            )
    return errors


def check_claim(claim: Mapping[str, Any], record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (rule, message) errors of one claim; empty when it passes. A ``limitation`` claim
    is exempt from rule (a) but may not cite an E or A region (spec card, 24 Nov)."""
    errors: list[tuple[str, str]] = []
    ctype, text = claim.get("claim_type"), str(claim.get("text", ""))
    ids = [str(i) for i in claim.get("evidence_ids") or []]
    if ctype not in prompts.CLAIM_TYPES:
        errors.append(("schema", f"claim_type {ctype!r} is not one of {list(prompts.CLAIM_TYPES)}"))
    known = entries(record)
    missing = [i for i in ids if i not in known]
    if ctype == "limitation" and _region_ids(ids):
        errors.append(("limitation", f"a limitation may not cite regions: {_region_ids(ids)}"))
    if not ids and ctype != "limitation":
        errors.append(("a", "the claim cites no evidence ID"))
    if missing:
        errors.append(("a", f"cited IDs not in the evidence: {missing}"))
    cited = [known[i] for i in ids if i in known]
    errors += _check_numbers(text, ids, cited, record)
    errors += _check_regions(ctype, text, ids, known)
    return errors


def _region_kind(claim: Mapping[str, Any]) -> str:
    ids = [str(i) for i in claim.get("evidence_ids") or []]
    if any(i.startswith("E") for i in ids):
        return "E"
    if any(i.startswith("A") for i in ids):
        return "A"
    return "other"


def _confidence_error(report: Mapping[str, Any], record: Mapping[str, Any]) -> str | None:
    """The top-level confidence must equal PRED[verdict] within the rule (b) tolerance (a
    percentage within 1 point)."""
    pred = record.get("prediction", {})
    verdict = report.get("verdict")
    conf = report.get("confidence")
    if verdict not in pred:
        return None  # the verdict error already covers it
    if isinstance(conf, bool) or not isinstance(conf, int | float):
        return f"the confidence {conf!r} is not a number"
    value = [(verdict, float(pred[verdict]))]
    if number_matches(float(conf), False, value) or number_matches(float(conf), True, value):
        return None
    return f"the confidence {conf} differs from the prediction for {verdict!r} ({pred[verdict]})"


def verify(report: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Per-claim errors of a parsed report, its passing claims, and the report-level errors:
    a verdict that differs from the evidence record's (the report may not overrule the
    detector) and a confidence that differs from PRED[verdict]."""
    checked = [{"claim": dict(c), "errors": check_claim(c, record)} for c in report["claims"]]
    mismatch = report.get("verdict") != record.get("verdict")
    return {
        "checked": checked,
        "passed": [c["claim"] for c in checked if not c["errors"]],
        "failed": [c for c in checked if c["errors"]],
        "verdict_error": (
            f"the report's verdict {report.get('verdict')!r} differs from the evidence verdict "
            f"{record.get('verdict')!r}"
            if mismatch
            else None
        ),
        "confidence_error": _confidence_error(report, record),
    }


def _needs_retry(r: Mapping[str, Any]) -> bool:
    cur = r["current"]
    return (
        r["parse_failed"]
        or cur is None
        or bool(cur["failed"])
        or cur["verdict_error"] is not None
        or cur["confidence_error"] is not None
    )


def _parse(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return prompts.parse_report(text), None
    except prompts.ReportError as e:
        return None, str(e)


def _first_result(uid: str, rec: Mapping[str, Any], text: str) -> dict[str, Any]:
    rep, err = _parse(text)
    checks = verify(rep, rec) if rep else None
    return {
        "uid": uid,
        "first_text": text,
        "first_report": rep,
        "first_parse_error": err,
        "first_checks": checks,
        "retries": 0,
        "rejections": [],
        "current_text": text,
        "current_report": rep,  # the last PARSED report and its checks
        "current": checks,
        "parse_failed": rep is None,
    }


def _errors_to_return(r: Mapping[str, Any]) -> list[str]:
    cur = r["current"]
    errs = ["the answer is not valid JSON in the schema"] if r["parse_failed"] else []
    if cur is not None:
        errs += [
            f"claim {c['claim'].get('text')!r}: {m}" for c in cur["failed"] for _, m in c["errors"]
        ]
        errs += [e for e in (cur["verdict_error"], cur["confidence_error"]) if e]
    return errs


def _retry_round(llm: LLM, todo: list[dict[str, Any]], by_uid: Mapping[str, Any]) -> None:
    retry = []
    for r in todo:
        errs = _errors_to_return(r)
        r["rejections"].append(errs)
        retry.append(prompts.retry_prompt(by_uid[r["uid"]], r["current_text"], errs))
    for r, text in zip(todo, llm.generate(retry), strict=True):
        rep, _ = _parse(text)
        r["retries"] += 1
        r["current_text"] = text
        r["parse_failed"] = rep is None
        if rep is not None:  # an unparseable retry keeps the last parsed report and its checks
            r["current_report"] = rep
            r["current"] = verify(rep, by_uid[r["uid"]])


def _finalize(r: dict[str, Any], record: Mapping[str, Any]) -> None:
    """The final report, from the last parsed report: its passing claims only. Whenever its
    verdict differs from the evidence verdict it is replaced (flagged ``verdict_replaced``); a
    confidence that still differs from PRED[verdict] is replaced too (``confidence_replaced``).
    No parsed report at all leaves ``final_report`` None."""
    rep = r.pop("current_report")
    current = r.pop("current")
    r.pop("current_text")
    r.pop("parse_failed")
    r["dropped_claims"] = len(current["failed"]) if current else None
    if rep is None:
        r.update(final_report=None, verdict_replaced=False, confidence_replaced=False)
        return
    final = {**rep, "claims": current["passed"]}
    r["verdict_replaced"] = final.get("verdict") != record["verdict"]
    if r["verdict_replaced"]:
        final["verdict"] = record["verdict"]
    expected = record["prediction"][final["verdict"]]
    r["confidence_replaced"] = r["verdict_replaced"] or current["confidence_error"] is not None
    if r["confidence_replaced"]:
        final["confidence"] = expected
    r["final_report"] = final


def run_grounded(
    llm: LLM, records: Sequence[tuple[str, Mapping[str, Any]]], *, max_retries: int = MAX_RETRIES
) -> list[dict[str, Any]]:
    """The grounded reports of ``records`` ((uid, evidence) pairs), batched: one first-pass call
    for all, then up to ``max_retries`` rounds for the reports with failing claims or no valid
    JSON. Each result keeps the first pass (text, report, checks) and the final report."""
    texts = llm.generate([prompts.grounded_prompt(r) for _, r in records])
    results = [_first_result(uid, rec, t) for (uid, rec), t in zip(records, texts, strict=True)]
    by_uid = dict(records)
    for _ in range(max_retries):
        todo = [r for r in results if _needs_retry(r)]
        if not todo:
            break
        _retry_round(llm, todo, by_uid)
    for r in results:
        _finalize(r, by_uid[r["uid"]])
    return results


def _tally_claims(checked: Iterable[Mapping[str, Any]], tally: Counter[str]) -> None:
    """Add one report's checked claims to ``tally``: ``claims`` / ``passed`` for the claims GR
    counts, ``limitation:passed|failed`` for limitation claims, and ``rule:<r>`` /
    ``kind:<r>:<E|A|...>`` per rejection."""
    for c in checked:
        ok = not c["errors"]
        if c["claim"].get("claim_type") == "limitation":
            tally["limitation:" + ("passed" if ok else "failed")] += 1
            continue
        tally["claims"] += 1
        tally["passed"] += int(ok)
        for rule in {e[0] for e in c["errors"]}:
            tally["rule:" + rule] += 1
            tally[f"kind:{rule}:{_region_kind(c['claim'])}"] += 1


def _sub(tally: Counter[str], prefix: str) -> dict[str, int]:
    return {k[len(prefix) :]: v for k, v in tally.items() if k.startswith(prefix)}


def first_pass_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """GR and HR-E on the first pass (before the verifier removes anything), with rejections
    counted per rule and per region kind (rule (c) for E regions, (c') for A regions). A first
    answer that is not valid JSON counts as one failed claim. ``limitation`` claims are left out
    of GR / HR-E and counted separately (spec card, 24 Nov)."""
    tally: Counter[str] = Counter()
    for r in results:
        checks = r["first_checks"]
        if checks is None:
            tally["unparsed"] += 1
            tally["claims"] += 1
            continue
        tally["verdict_mismatch"] += int(checks["verdict_error"] is not None)
        tally["confidence_mismatch"] += int(checks["confidence_error"] is not None)
        _tally_claims(checks["checked"], tally)
    total, passed = tally["claims"], tally["passed"]
    gr = passed / total if total else None
    return {
        "claims": total,
        "passed": passed,
        "gr": gr,
        "hr_e": None if gr is None else 1 - gr,
        "unparsed_first_answers": tally["unparsed"],
        "verdict_mismatch_first_answers": tally["verdict_mismatch"],
        "confidence_mismatch_first_answers": tally["confidence_mismatch"],
        "limitation_claims": _sub(tally, "limitation:"),
        "rejections_by_rule": _sub(tally, "rule:"),
        "rejections_by_rule_and_region": _sub(tally, "kind:"),
    }


def box_inside_fraction(box: Sequence[float], mask: np.ndarray) -> float:
    """Share of a pixel box [x, y, w, h] that lies inside a boolean mask, over the **whole**
    box: any part of the box outside the image counts as outside the mask (0 if empty)."""
    h, w = mask.shape[:2]
    bx0, by0 = int(round(box[0])), int(round(box[1]))
    bx1, by1 = int(round(box[0] + box[2])), int(round(box[1] + box[3]))
    area = (bx1 - bx0) * (by1 - by0)
    x0, y0, x1, y1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
    if area <= 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.asarray(mask[y0:y1, x0:x1], dtype=bool).sum()) / area


def _claim_boxes(claim: Mapping[str, Any], known: Mapping[str, Mapping[str, Any]]) -> list:
    """The pixel boxes an edited-region claim is scored on. Attribution claims (A regions of a
    fully generated verdict) state where the model's evidence lies, not where an edit is, and
    have no ground truth, so they are not checkable; neither are global claims."""
    if claim.get("claim_type") not in ("edited_region", None):
        return []
    if claim.get("box") is not None:
        return [claim["box"]]
    return [known[i]["bbox"] for i in claim.get("evidence_ids") or [] if "bbox" in known.get(i, {})]


def hr_gt_counts(
    report: Mapping[str, Any] | None,
    true_label: str,
    gt_mask: np.ndarray | None,
    record: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Checkable and wrong claims of one report against the ground truth: the verdict (wrong
    when it differs from the true class) and every edited-region claim (wrong when < 10% of its
    box is inside the true edit mask; no mask = nothing inside; attribution claims are not
    checkable). EviGround's region boxes come from the
    cited E/A entries of ``record``; the image baseline gives a ``box`` per claim. Claims about a
    property with no ground truth (global evidence) are not checkable."""
    if report is None:
        return {"checkable": 1, "wrong": 1}
    known = entries(record) if record else {}
    boxes = [b for c in report.get("claims", []) for b in _claim_boxes(c, known)]
    inside = [0.0 if gt_mask is None else box_inside_fraction(b, gt_mask) for b in boxes]
    verdict_wrong = int(report.get("verdict") != TRUE_VERDICT[true_label])
    return {
        "checkable": 1 + len(boxes),
        "wrong": verdict_wrong + sum(f < BOX_INSIDE_MIN for f in inside),
    }


def hr_gt_summary(
    rows: Iterable[
        tuple[Mapping[str, Any] | None, str, np.ndarray | None, Mapping[str, Any] | None]
    ],
) -> dict[str, Any]:
    """HR-GT and verdict accuracy over (report, true_label, gt_mask, record) rows; ``record`` is
    None for the image baseline (its boxes come with the claims)."""
    checkable = wrong = right_verdicts = images = 0
    for report, label, mask, record in rows:
        c = hr_gt_counts(report, label, mask, record)
        checkable += c["checkable"]
        wrong += c["wrong"]
        images += 1
        right_verdicts += int(report is not None and report.get("verdict") == TRUE_VERDICT[label])
    return {
        "images": images,
        "checkable_claims": checkable,
        "wrong_claims": wrong,
        "hr_gt": wrong / checkable if checkable else None,
        "verdict_accuracy": right_verdicts / images if images else None,
    }
