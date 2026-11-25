"""CP6 analysis after Phase D (Dhanush's review, 25 Nov 2026): counts that need no GPU.

- :func:`rule_counts`: failing claims of a set of reports, by verifier rule (the template bug,
  as run and fixed);
- :func:`final_check`: CP6 criterion 1, every claim of every final report passes all rules;
- :func:`verdict_confusion`: true class x stated verdict (3 x 3, plus unparsed answers);
- :func:`boxes_by_claim_type`: the image baseline's boxed and unboxed claims per claim type;
- :func:`read_entry`: one of the 30 reports to read (criterion 2), with every claim's errors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from eviground.explain import prompts, variants, verifier

UNPARSED = "unparsed"


def _checked(report: Mapping[str, Any] | None, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report is None:
        return []
    return [
        {"claim": dict(c), "errors": [list(e) for e in verifier.check_claim(c, record)]}
        for c in report.get("claims", [])
    ]


def rule_counts(
    pairs: Iterable[tuple[Mapping[str, Any] | None, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Claims, failing claims and reports with a failing claim over (report, record) pairs,
    with the failing claims counted per rule (a claim breaking two rules counts under both) and
    per claim type."""
    tally: Counter[str] = Counter()
    by_rule: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    for report, record in pairs:
        tally["reports"] += 1
        checked = _checked(report, record)
        failing = [c for c in checked if c["errors"]]
        tally["claims"] += len(checked)
        tally["failing_claims"] += len(failing)
        tally["reports_with_a_failing_claim"] += int(bool(failing))
        for c in failing:
            by_rule.update({e[0] for e in c["errors"]})
            by_type[str(c["claim"].get("claim_type"))] += 1
    return {
        **{k: tally[k] for k in ("reports", "claims", "failing_claims")},
        "reports_passing": tally["reports"] - tally["reports_with_a_failing_claim"],
        "reports_with_a_failing_claim": tally["reports_with_a_failing_claim"],
        "failing_claims_by_rule": dict(sorted(by_rule.items())),
        "failing_claims_by_type": dict(sorted(by_type.items())),
    }


def final_check(
    results: Iterable[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """CP6 criterion 1 on the stored grounded results: every claim of every final report passes
    all verifier rules (so cites existing evidence IDs), and the final verdict and confidence
    agree with the evidence record. ``passes`` is True only if nothing fails."""
    results = list(results)
    counts = rule_counts((r["final_report"], records[r["uid"]]) for r in results)
    no_final = sum(r["final_report"] is None for r in results)
    verdict_bad = conf_bad = 0
    for r in results:
        rep = r["final_report"]
        if rep is None:
            continue
        rec = records[r["uid"]]
        verdict_bad += int(rep.get("verdict") != rec["verdict"])
        conf_bad += int(verifier._confidence_error(rep, rec) is not None)
    return {
        **counts,
        "reports_without_a_final_report": no_final,
        "final_verdict_mismatches": verdict_bad,
        "final_confidence_mismatches": conf_bad,
        "passes": not (counts["failing_claims"] or no_final or verdict_bad or conf_bad),
    }


def verdict_confusion(
    rows: Iterable[tuple[Mapping[str, Any] | None, str]],
) -> dict[str, dict[str, int]]:
    """True class (as the report's verdict name) -> stated verdict -> count, over (report,
    true_label) rows; true_label is 'real', 'generated' or 'edited'. A missing report or a
    verdict outside the schema counts as ``unparsed``."""
    cols = (*prompts.VERDICTS, UNPARSED)
    out = {t: dict.fromkeys(cols, 0) for t in prompts.VERDICTS}
    for report, label in rows:
        said = (report or {}).get("verdict")
        out[verifier.TRUE_VERDICT[label]][said if said in prompts.VERDICTS else UNPARSED] += 1
    return out


def boxes_by_claim_type(reports: Iterable[Mapping[str, Any] | None]) -> dict[str, dict[str, int]]:
    """Claim type -> {'with_box', 'without_box'} over the image baseline's raw reports."""
    out: dict[str, Counter[str]] = {}
    for rep in reports:
        for c in (rep or {}).get("claims", []):
            kind = str(c.get("claim_type"))
            out.setdefault(kind, Counter())["with_box" if "box" in c else "without_box"] += 1
    return {k: {"with_box": v["with_box"], "without_box": v["without_box"]}
            for k, v in sorted(out.items())}  # fmt: skip


def read_entry(
    uid: str,
    record: Mapping[str, Any],
    grounded: Mapping[str, Any],
    true_label: str,
    image: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One report to read beside its image (CP6 criterion 2): the evidence record, the first
    pass and the final report with each claim's verifier errors, the fixed template, and the
    image baseline's raw report when the image is in P6_mllm."""
    return {
        "uid": uid,
        "true_label": true_label,
        "true_verdict": verifier.TRUE_VERDICT[true_label],
        "evidence": record,
        "retries": grounded.get("retries"),
        "first_pass": {
            "verdict": (grounded.get("first_report") or {}).get("verdict"),
            "claims": _checked(grounded.get("first_report"), record),
        },
        "final": {
            "verdict": (grounded.get("final_report") or {}).get("verdict"),
            "confidence": (grounded.get("final_report") or {}).get("confidence"),
            "claims": _checked(grounded.get("final_report"), record),
        },
        "template": _template_entry(record),
        "image_baseline": None if image is None else image.get("report"),
    }


def _template_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"claims": _checked(variants.template_report(record), record)}
