"""A5 explanation variants (spec section 9, ablation 5; Dhanush, 24 Nov 2026).

1. **Grounded LLM with the verifier**: :func:`verifier.run_grounded`, final reports.
2. **No verifier**: the same run's first pass (stored next to the final report).
3. **Fixed template, no LLM**: :func:`template_report`, deterministic from the evidence record.
4. **LLM shown the image**: Qwen2-VL-7B-Instruct (AWQ) on P6_mllm, same report schema with a pixel
   box per region claim (:func:`image_reports`). It gets no evidence record, so GR / HR-E are
   not defined for it; HR-GT and verdict accuracy are the cross-system comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eviground.explain import prompts


def template_report(record: Mapping[str, Any]) -> dict[str, Any]:
    """A report written by fixed rules from the evidence record; every number is copied from
    the cited entry, so it passes the verifier by construction (checked in the tests)."""
    pred, top = record["prediction"], record["verdict"]
    g1 = next(g for g in record["global"] if g["id"] == "G1")
    claims: list[dict[str, Any]] = [
        {
            "claim_type": "verdict",
            "text": f"The detector gives this verdict a probability of {pred[top]}.",
            "evidence_ids": [prompts.PRED_ID],
        },
        {
            "claim_type": "global",
            "text": f"The decision relies on the semantic branch with weight {g1['semantic']} "
            f"and on the trace branch with weight {g1['trace']}.",
            "evidence_ids": ["G1"],
        },
    ]
    for reg in record.get("regions", []):
        if reg["id"].startswith("E"):
            claims.append(
                {
                    "claim_type": "edited_region",
                    "text": f"Region {reg['id']} is marked as edited; it covers {reg['area_pct']}% "
                    f"of the image with mean mask confidence {reg['mask_conf']}.",
                    "evidence_ids": [reg["id"]],
                }
            )
        else:
            claims.append(
                {
                    "claim_type": "attribution",
                    "text": f"The model's evidence concentrates in {reg['id']}, "
                    f"which holds {reg['attr_mass']} of the attribution mass.",
                    "evidence_ids": [reg["id"]],
                }
            )
    return {
        "verdict": top,
        "confidence": pred[top],
        "claims": claims,
        "limitations": "Written from the detector's evidence record by fixed rules.",
    }


def image_reports(llm: Any, images: Sequence[Any]) -> list[dict[str, Any]]:
    """The image baseline: one prompt per image (the same schema, a box per region claim), run
    through ``llm.generate_with_images``. Unparseable answers are kept as text with
    ``report=None`` (they count as a wrong verdict in HR-GT)."""
    texts = llm.generate_with_images([prompts.IMAGE_PROMPT] * len(images), images)
    out = []
    for text in texts:
        try:
            rep, err = prompts.parse_report(text), None
        except prompts.ReportError as e:
            rep, err = None, str(e)
        out.append({"text": text, "report": rep, "parse_error": err})
    return out
