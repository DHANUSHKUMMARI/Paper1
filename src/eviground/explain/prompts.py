"""M6 prompts and the report schema (spec section 8, M6; verifier thresholds pre-registered on
24 Nov 2026, spec card for Dhanush).

The grounded prompt is built from the M5 evidence record **only**: no image, no pixel values, no
file path and no uid. The image baseline (Qwen2-VL) gets the same schema, with a pixel box for
each region claim so that HR-GT can score it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from eviground.explain.llm import EVIDENCE_TAG, JSON_BLOCK

CLAIM_TYPES = ("verdict", "edited_region", "attribution", "global", "limitation")
VERDICTS = ("real", "fully_generated", "locally_edited")
EVIDENCE_KEYS = ("prediction", "verdict", "global", "regions")
PRED_ID = "PRED"  # the evidence ID of the prediction block

SCHEMA = (
    '{"verdict": one of "real" | "fully_generated" | "locally_edited", "confidence": number, '
    '"claims": [{"claim_type": one of ' + " | ".join(f'"{c}"' for c in CLAIM_TYPES) + ", "
    '"text": string, "evidence_ids": [string, ...]}], "limitations": string}'
)

GROUNDED = f"""You are a forensic analyst writing a short report for a non-expert.
You do NOT see the image. You see only the detector's evidence record below.

Rules:
1. Every claim must cite at least one evidence ID from the record: {PRED_ID} for the prediction,
   G1-G3 for global evidence, E1... for edited regions, A1... for attribution regions.
2. Quote numbers only as they appear in the cited evidence.
3. Claims about E regions use claim_type "edited_region". Claims about A regions may only say
   where the model's evidence concentrates (claim_type "attribution"); never call an A region
   edited or manipulated.
4. Answer with one JSON object and nothing else, in this schema:
{SCHEMA}

{{evidence_line}}
"""

# Qwen2-VL's native grounding format: (x1, y1, x2, y2) on a 0-1000 grid over the image
# (Dhanush, 24 Nov); converted to pixel [x, y, w, h] per image before HR-GT (variants.py).
BOX_GRID = 1000
IMAGE_SCHEMA = SCHEMA.replace(
    '"evidence_ids": [string, ...]}',
    '"evidence_ids": [], "box": [x1, y1, x2, y2] (region claims only)}',
)

IMAGE_PROMPT = f"""You are a forensic analyst writing a short report for a non-expert.
Look at the image. Decide whether it is a real photograph, fully AI-generated, or a real or
generated image with a locally edited region. For every claim about a region, give its box as
[x1, y1, x2, y2]: the top-left and bottom-right corners on a 0-{BOX_GRID} grid over the image
(0 = left or top edge, {BOX_GRID} = right or bottom edge). Answer with one JSON object and
nothing else, in this schema:
{IMAGE_SCHEMA}
"""


class ReportError(ValueError):
    """The model's answer is not a report in the schema."""


def evidence_only(record: Mapping[str, Any]) -> dict[str, Any]:
    """The evidence record's own keys, nothing else (a bundle's uid or paths never pass)."""
    return {k: record[k] for k in EVIDENCE_KEYS if k in record}


def evidence_line(record: Mapping[str, Any]) -> str:
    return EVIDENCE_TAG + json.dumps(evidence_only(record), sort_keys=True, separators=(",", ":"))


def grounded_prompt(record: Mapping[str, Any]) -> str:
    """The grounded LLM's prompt for one image, from its evidence record only."""
    return GROUNDED.replace("{evidence_line}", evidence_line(record))


def retry_prompt(record: Mapping[str, Any], report_text: str, errors: Sequence[str]) -> str:
    """The same prompt, the previous answer and the verifier's errors (spec: failing claims are
    returned to the LLM with the error)."""
    return (
        grounded_prompt(record)
        + "\nYour previous answer:\n"
        + report_text.strip()
        + "\nVERIFIER ERRORS (fix or remove these claims, keep the rest):\n"
        + "\n".join(f"- {e}" for e in errors)
        + "\n"
    )


def parse_report(text: str) -> dict[str, Any]:
    """The JSON object in a model's answer, checked against the schema's shape."""
    match = JSON_BLOCK.search(text or "")
    if not match:
        raise ReportError("no JSON object in the answer")
    try:
        rep = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ReportError(f"invalid JSON: {e}") from e
    if not isinstance(rep, dict) or not isinstance(rep.get("claims"), list):
        raise ReportError("the answer has no claims list")
    if rep.get("verdict") not in VERDICTS:
        raise ReportError(f"unknown verdict {rep.get('verdict')!r}")
    for c in rep["claims"]:
        if not isinstance(c, dict) or not isinstance(c.get("text"), str):
            raise ReportError("a claim is not an object with text")
        c.setdefault("evidence_ids", [])
    return rep
