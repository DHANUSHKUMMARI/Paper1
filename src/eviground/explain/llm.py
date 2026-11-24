"""M6 language models behind one interface (spec section 8, M6; Dhanush, 24 Nov 2026).

- :class:`MockLLM`: deterministic and schema-valid, built from the evidence record in the prompt,
  with injectable faults (a missing ID, a wrong number, a claim on a weak region, an A region
  called edited, broken JSON). It repairs a fault when the retry prompt names it, so the verifier's
  retry path can be tested without a GPU.
- :class:`VLLMBackend`: vLLM with the official AWQ 4-bit Qwen checkpoints, temperature 0, batched,
  float16 (the T4 has no bfloat16; the same dtype on the L4 keeps both runtimes alike). vLLM is
  imported only when the backend is built, so the sandbox and CI never need it.

Neither reads images or the detector: the grounded LLM gets text built from the evidence record
only (:mod:`eviground.explain.prompts`); the image baseline gets the image through
:meth:`VLLMBackend.generate_with_images`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

# Pinned Hugging Face revisions (commit SHAs, read 24 Nov 2026).
QWEN25_AWQ = ("Qwen/Qwen2.5-7B-Instruct-AWQ", "b25037543e9394b818fdfca67ab2a00ecc7dd641")
QWEN2VL_AWQ = ("Qwen/Qwen2-VL-7B-Instruct-AWQ", "6ec2560b0afc3a618d4acc9b8e2967d1642f463d")
MAX_TOKENS = 768
FAULTS = ("bad_id", "bad_number", "weak_region", "a_edited", "bad_json", "bad_verdict")
EVIDENCE_TAG = "EVIDENCE:"


class LLM(Protocol):
    def generate(self, prompts: Sequence[str]) -> list[str]: ...


def evidence_from_prompt(prompt: str) -> dict[str, Any]:
    """The evidence JSON that :mod:`prompts` places after ``EVIDENCE:`` on its own line."""
    line = next(ln for ln in prompt.splitlines() if ln.startswith(EVIDENCE_TAG))
    return json.loads(line[len(EVIDENCE_TAG) :])


class MockLLM:
    """A deterministic stand-in for the 7B model. ``faults`` maps a prompt index (first pass, in
    call order) to a fault name; a retry prompt that quotes the verifier's error for that fault
    gets a repaired report."""

    def __init__(
        self,
        faults: Mapping[int, str] | None = None,
        *,
        repair: bool = True,
        retry_fault: str | None = None,
    ):
        self.faults = dict(faults or {})
        self.repair = repair
        self.retry_fault = retry_fault  # a fault every retry answer has (e.g. "bad_json")
        self.calls: list[list[str]] = []
        self._first = 0
        self._last_fault: str | None = None

    def generate(self, prompts: Sequence[str]) -> list[str]:
        self.calls.append(list(prompts))
        out = []
        for p in prompts:
            retry = "VERIFIER ERRORS" in p
            fault = None
            if not retry:
                fault = self.faults.get(self._first)
                self._last_fault = fault or self._last_fault
                self._first += 1
            elif self.retry_fault:
                fault = self.retry_fault
            elif not self.repair:
                fault = self._last_fault  # repeat the mistake on every retry
            out.append(self._report(evidence_from_prompt(p), fault))
        return out

    @staticmethod
    def _report(ev: Mapping[str, Any], fault: str | None) -> str:
        if fault == "bad_json":
            return "The image looks edited {not json"
        pred = ev["prediction"]
        top = ev["verdict"]
        claims: list[dict[str, Any]] = [
            {
                "claim_type": "verdict",
                "text": f"The detector's probability for this verdict is {pred[top]:.2f}.",
                "evidence_ids": ["PRED"],
            },
            {
                "claim_type": "global",
                "text": f"The semantic branch weight is {ev['global'][0]['semantic']:.2f}.",
                "evidence_ids": ["G1"],
            },
        ]
        for reg in ev.get("regions", []):
            if reg["id"].startswith("E"):
                claims.append(
                    {
                        "claim_type": "edited_region",
                        "text": f"Region {reg['id']} covers {reg['area_pct']:.1f}% of the image.",
                        "evidence_ids": [reg["id"]],
                    }
                )
            else:
                claims.append(
                    {
                        "claim_type": "attribution",
                        "text": f"The model's evidence concentrates in {reg['id']}.",
                        "evidence_ids": [reg["id"]],
                    }
                )
        if fault == "bad_id":
            claims.append(
                {"claim_type": "global", "text": "Noise is uneven.", "evidence_ids": ["G9"]}
            )
        elif fault == "bad_number":
            claims[1]["text"] = "The semantic branch weight is 0.01."
            if abs(ev["global"][0]["semantic"] - 0.01) <= 0.02:
                claims[1]["text"] = "The semantic branch weight is 0.95."
        elif fault == "weak_region":
            claims.append(
                {"claim_type": "edited_region", "text": "E1 is edited.", "evidence_ids": ["E1"]}
            )
        elif fault == "a_edited":
            claims.append(
                {"claim_type": "edited_region", "text": "A1 was edited.", "evidence_ids": ["A1"]}
            )
        if fault == "bad_verdict":
            top = "real" if top != "real" else "fully_generated"
        report = {
            "verdict": top,
            "confidence": pred[top],
            "claims": claims,
            "limitations": "Based only on the detector's evidence record.",
        }
        return json.dumps(report)


class VLLMBackend:
    """vLLM, AWQ 4-bit, temperature 0, float16, batched; ``model`` is (repo, revision)."""

    def __init__(
        self,
        model: tuple[str, str] = QWEN25_AWQ,
        *,
        max_model_len: int = 4096,
        max_tokens: int = MAX_TOKENS,
        gpu_memory_utilization: float = 0.85,
        download_dir: str | None = None,
        limit_mm_per_prompt: Mapping[str, int] | None = None,
    ):
        # vLLM is a GPU-only dependency, imported on use
        import vllm  # noqa: PLC0415

        self.repo, self.revision = model
        self._vllm = vllm
        kwargs: dict[str, Any] = {
            "model": self.repo,
            "revision": self.revision,
            "tokenizer_revision": self.revision,
            "quantization": "awq",
            "dtype": "float16",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": 0,
        }
        if download_dir:
            kwargs["download_dir"] = download_dir
        if limit_mm_per_prompt:
            kwargs["limit_mm_per_prompt"] = dict(limit_mm_per_prompt)
        self.engine = vllm.LLM(**kwargs)
        self.params = vllm.SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)

    def _chat(self, messages: list[list[dict[str, Any]]]) -> list[str]:
        outs = self.engine.chat(messages, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return self._chat([[{"role": "user", "content": p}] for p in prompts])

    def generate_with_images(self, prompts: Sequence[str], images: Sequence[Any]) -> list[str]:
        """The image baseline: each prompt with its PIL image (Qwen2-VL), sent as a PNG data URL
        in the chat message (only this baseline ever sees pixels)."""
        msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_data_url(img)}},
                        {"type": "text", "text": p},
                    ],
                }
            ]
            for p, img in zip(prompts, images, strict=True)
        ]
        return self._chat(msgs)

    def close(self) -> None:
        """Free the GPU before the next model is loaded in the same session."""
        import gc  # noqa: PLC0415

        del self.engine
        try:
            from vllm.distributed.parallel_state import (  # noqa: PLC0415
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except ImportError:  # older or newer vLLM layouts
            pass
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except ImportError:
            pass


def png_data_url(image: Any) -> str:
    """A PIL image or an (H, W, 3) uint8 array as a base64 PNG data URL."""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class MockVL:
    """A stand-in for the image baseline: a schema-valid report with one boxed region claim."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_with_images(self, prompts: Sequence[str], images: Sequence[Any]) -> list[str]:
        if len(prompts) != len(images):
            raise ValueError("one prompt per image")
        self.calls += 1
        out = []
        for _ in images:
            rep = {
                "verdict": "locally_edited",
                "confidence": 0.7,
                "claims": [
                    {
                        "claim_type": "edited_region",
                        "text": "The upper left looks altered.",
                        "evidence_ids": [],
                        "box": [0, 0, 500, 500],  # Qwen2-VL's 0-1000 grid
                    }
                ],
                "limitations": "",
            }
            out.append(json.dumps(rep))
        return out


JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
