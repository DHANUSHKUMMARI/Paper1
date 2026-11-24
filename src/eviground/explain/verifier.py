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


def check_claim(claim: Mapping[str, Any], record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (rule, message) errors of one claim; empty when it passes."""
    errors: list[tuple[str, str]] = []
    ctype, text = claim.get("claim_type"), str(claim.get("text", ""))
    ids = [str(i) for i in claim.get("evidence_ids") or []]
    if ctype not in prompts.CLAIM_TYPES:
        errors.append(("schema", f"claim_type {ctype!r} is not one of {list(prompts.CLAIM_TYPES)}"))
    known = entries(record)
    missing = [i for i in ids if i not in known]
    if not ids:
        errors.append(("a", "the claim cites no evidence ID"))
    if missing:
        errors.append(("a", f"cited IDs not in the evidence: {missing}"))
    cited = [known[i] for i in ids if i in known]
    scrubbed = re.sub(r"\b[PGEA]\w*\d+\b", " ", text)  # IDs like E1, G2, PRED are not numbers
    values = [kv for e in cited for kv in _values(e)]
    for m in NUMBER.finditer(scrubbed):
        q, pct = float(m.group(1)), bool(m.group(2))
        if not number_matches(q, pct, values):
            errors.append(("b", f"the number {m.group(0).strip()} matches no value of {ids}"))
    for i in ids:
        if i.startswith("E") and i in known and float(known[i].get("mask_conf", 0.0)) < TAU_E:
            errors.append(("c", f"{i} has mask_conf {known[i].get('mask_conf')} < {TAU_E}"))
        if i.startswith("A") and i in known and (ctype != "attribution" or EDIT_WORDS.search(text)):
            errors.append(
                (
                    "c_prime",
                    f"{i} is an attribution region; it may only be described as "
                    "where the model's evidence concentrates, never as edited",
                )
            )
    return errors


def _region_kind(claim: Mapping[str, Any]) -> str:
    ids = [str(i) for i in claim.get("evidence_ids") or []]
    if any(i.startswith("E") for i in ids):
        return "E"
    if any(i.startswith("A") for i in ids):
        return "A"
    return "other"


def verify(report: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Per-claim errors of a parsed report, and its passing claims."""
    checked = [{"claim": dict(c), "errors": check_claim(c, record)} for c in report["claims"]]
    return {
        "checked": checked,
        "passed": [c["claim"] for c in checked if not c["errors"]],
        "failed": [c for c in checked if c["errors"]],
    }


def _parse(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return prompts.parse_report(text), None
    except prompts.ReportError as e:
        return None, str(e)


def run_grounded(
    llm: LLM, records: Sequence[tuple[str, Mapping[str, Any]]], *, max_retries: int = MAX_RETRIES
) -> list[dict[str, Any]]:
    """The grounded reports of ``records`` ((uid, evidence) pairs), batched: one first-pass call
    for all, then up to ``max_retries`` rounds for the reports with failing claims or no valid
    JSON. Each result keeps the first pass (text, report, checks) and the final report."""
    prompts_1 = [prompts.grounded_prompt(r) for _, r in records]
    texts = llm.generate(prompts_1)
    results = []
    for (uid, rec), text in zip(records, texts, strict=True):
        rep, err = _parse(text)
        res: dict[str, Any] = {
            "uid": uid,
            "first_text": text,
            "first_report": rep,
            "first_parse_error": err,
            "first_checks": verify(rep, rec) if rep else None,
            "retries": 0,
            "rejections": [],
        }
        res["current_text"], res["current"] = text, res["first_checks"]
        results.append(res)
    by_uid = {uid: rec for uid, rec in records}
    for _ in range(max_retries):
        todo = [r for r in results if r["current"] is None or r["current"]["failed"]]
        if not todo:
            break
        retry = []
        for r in todo:
            errs = (
                ["the answer is not valid JSON in the schema"]
                if r["current"] is None
                else [
                    f"claim {c['claim'].get('text')!r}: {m}"
                    for c in r["current"]["failed"]
                    for _, m in c["errors"]
                ]
            )
            r["rejections"].append(errs)
            retry.append(prompts.retry_prompt(by_uid[r["uid"]], r["current_text"], errs))
        for r, text in zip(todo, llm.generate(retry), strict=True):
            rep, _ = _parse(text)
            r["retries"] += 1
            r["current_text"] = text
            r["current"] = verify(rep, by_uid[r["uid"]]) if rep else None
            if rep:
                r["current_report"] = rep
    for r in results:
        rep = r.get("current_report") or r["first_report"]
        final_claims = r["current"]["passed"] if r["current"] else []
        r["final_report"] = None if rep is None else {**rep, "claims": final_claims}
        r["dropped_claims"] = len(r["current"]["failed"]) if r["current"] else None
        del r["current"], r["current_text"]
        r.pop("current_report", None)
    return results


def first_pass_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """GR and HR-E on the first pass (before the verifier removes anything), with rejections
    counted per rule and per region kind (rule (c) for E regions, (c') for A regions). A first
    answer that is not valid JSON counts as one failed claim."""
    total = passed = 0
    by_rule: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    unparsed = 0
    for r in results:
        checks = r["first_checks"]
        if checks is None:
            unparsed += 1
            total += 1
            continue
        for c in checks["checked"]:
            total += 1
            if not c["errors"]:
                passed += 1
                continue
            for rule in {e[0] for e in c["errors"]}:
                by_rule[rule] += 1
                by_kind[f"{rule}:{_region_kind(c['claim'])}"] += 1
    gr = passed / total if total else None
    return {
        "claims": total,
        "passed": passed,
        "gr": gr,
        "hr_e": None if gr is None else 1 - gr,
        "unparsed_first_answers": unparsed,
        "rejections_by_rule": dict(by_rule),
        "rejections_by_rule_and_region": dict(by_kind),
    }


def box_inside_fraction(box: Sequence[float], mask: np.ndarray) -> float:
    """Share of a pixel box [x, y, w, h] that lies inside a boolean mask (0 if empty)."""
    h, w = mask.shape[:2]
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(w, int(round(box[0] + box[2]))), min(h, int(round(box[1] + box[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.asarray(mask[y0:y1, x0:x1], dtype=bool).mean())


def hr_gt_counts(
    report: Mapping[str, Any] | None,
    true_label: str,
    gt_mask: np.ndarray | None,
    record: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Checkable and wrong claims of one report against the ground truth: the verdict (wrong
    when it differs from the true class) and every region claim (wrong when < 10% of its box is
    inside the true edit mask; no mask = nothing inside). EviGround's region boxes come from the
    cited E/A entries of ``record``; the image baseline gives a ``box`` per claim. Claims about a
    property with no ground truth (global evidence) are not checkable."""
    if report is None:
        return {"checkable": 1, "wrong": 1}
    checkable, wrong = 1, int(report.get("verdict") != TRUE_VERDICT[true_label])
    known = entries(record) if record else {}
    for c in report.get("claims", []):
        boxes = []
        if c.get("box") is not None:
            boxes = [c["box"]]
        else:
            boxes = [
                known[i]["bbox"]
                for i in c.get("evidence_ids") or []
                if i in known and "bbox" in known[i]
            ]
        if not boxes or c.get("claim_type") not in ("edited_region", "attribution", None):
            continue
        for box in boxes:
            checkable += 1
            inside = 0.0 if gt_mask is None else box_inside_fraction(box, gt_mask)
            wrong += int(inside < BOX_INSIDE_MIN)
    return {"checkable": checkable, "wrong": wrong}


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
