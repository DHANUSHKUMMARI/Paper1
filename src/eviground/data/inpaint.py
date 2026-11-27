"""Inpainting backends for the CP8 Claims-Edit set (Dhanush, 27 Nov 2026).

- :class:`MockInpainter`: deterministic and CPU-only, for tests and the notebook dry run;
- :class:`KolorsInpainter`: Kolors-Inpainting (Hugging Face ``Kwai-Kolors/Kolors-Inpainting``
  @ :data:`KOLORS_WEIGHTS_REV`, Apache-2.0) through the pipeline of the Kolors repository
  (``Kwai-Kolors/Kolors`` @ :data:`KOLORS_REPO_REV`, installed with ``--no-deps``), as the repo's
  ``inpainting/sample_inpainting.py`` builds it, in fp16 with model CPU offload so that it fits a
  T4. Imported lazily, so the sandbox and CI never need it.

Every backend maps (image (H, W, 3) uint8, mask (H, W) bool, prompt, seed) to a generated image
of the same shape; :func:`claims_edit.composite` then keeps only the pixels inside the mask.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

KOLORS_WEIGHTS = "Kwai-Kolors/Kolors-Inpainting"
KOLORS_WEIGHTS_REV = "3bb98b424c9a17c09bf9e4fa1d53ce4f5aa32dc6"
KOLORS_REPO = "https://github.com/Kwai-Kolors/Kolors"
KOLORS_REPO_REV = "038818d244ed103056abd10f429729a26af4d239"
KOLORS_LONG_SIDE = 1024  # Kolors' native size; sides rounded to a multiple of 64
TRAINING_EDITORS = ("SDXL", "FLUX", "SD 3.5", "SD 2", "HunyuanDiT", "Kandinsky 2.2")


class Inpainter(Protocol):
    name: str

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str, *, negative: str,
                seed: int, steps: int, guidance: float) -> np.ndarray: ...  # fmt: skip


def assert_not_training_editor(name: str) -> str:
    """CP8 pass rule (spec): the inpainting model is not one of the unified model's training
    editors. Returns a log line."""
    low = name.lower()
    hits = [e for e in TRAINING_EDITORS if e.lower() in low]
    if hits:
        raise ValueError(f"{name} is a training editor ({hits}): not allowed for CP8")
    return f"inpainting model {name}: not a training editor ({', '.join(TRAINING_EDITORS)})"


class MockInpainter:
    """Deterministic stand-in: inside the mask, the image darkened and textured by a seeded
    noise field (visible, so the drop rule keeps it)."""

    name = "mock"

    def __init__(self) -> None:
        self.calls = 0
        self.last_call: dict | None = None  # the request of the latest call (tests read it)

    def inpaint(self, image, mask, prompt, *, negative, seed, steps, guidance):
        self.calls += 1
        self.last_call = {"prompt": prompt, "negative": negative, "seed": seed,
                          "steps": steps, "guidance": guidance}  # fmt: skip
        rng = np.random.default_rng(seed)
        noise = rng.integers(0, 60, size=image.shape, dtype=np.int16)
        edited = np.clip(image.astype(np.int16) // 2 + noise, 0, 255).astype(np.uint8)
        inside = np.asarray(mask, bool)
        if edited.ndim == 3:
            inside = inside[..., None]
        return np.where(inside, edited, image)


def _fit(h: int, w: int, long_side: int = KOLORS_LONG_SIDE) -> tuple[int, int]:
    """(height, width) with the long side ``long_side`` and both sides multiples of 64."""
    s = long_side / max(h, w)
    return max(64, round(h * s / 64) * 64), max(64, round(w * s / 64) * 64)


class KolorsInpainter:
    """Kolors-Inpainting, fp16, model CPU offload (T4). ``weights_dir``: the pinned weights
    (``snapshot_download``); the pinned repo must be importable (``kolors``)."""

    name = f"Kolors-Inpainting@{KOLORS_WEIGHTS_REV[:12]} (repo @{KOLORS_REPO_REV[:12]})"

    def __init__(self, weights_dir: Path | str, device: str = "cuda") -> None:
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
        from kolors.models.modeling_chatglm import ChatGLMModel
        from kolors.models.tokenization_chatglm import ChatGLMTokenizer
        from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import (
            StableDiffusionXLInpaintPipeline,
        )

        d = Path(weights_dir)
        fp16 = torch.float16
        self.pipe = StableDiffusionXLInpaintPipeline(
            vae=AutoencoderKL.from_pretrained(d / "vae").to(fp16),
            text_encoder=ChatGLMModel.from_pretrained(d / "text_encoder", torch_dtype=fp16).half(),
            tokenizer=ChatGLMTokenizer.from_pretrained(d / "text_encoder"),
            unet=UNet2DConditionModel.from_pretrained(d / "unet").to(fp16),
            scheduler=EulerDiscreteScheduler.from_pretrained(d / "scheduler"),
        )
        self.pipe.enable_model_cpu_offload()  # the 6B text encoder and the UNet do not fit together
        self.pipe.enable_attention_slicing()
        self.torch = torch
        self.device = device
        self.seconds: list[float] = []

    def inpaint(self, image, mask, prompt, *, negative, seed, steps, guidance):
        from PIL import Image

        h, w = image.shape[:2]
        th, tw = _fit(h, w)
        img = Image.fromarray(image).resize((tw, th), Image.LANCZOS)
        msk = Image.fromarray(np.asarray(mask, np.uint8) * 255).resize((tw, th), Image.NEAREST)
        gen = self.torch.Generator(device="cpu").manual_seed(int(seed))
        t = time.time()
        out = self.pipe(prompt=prompt, image=img, mask_image=msk.convert("RGB"), height=th,
                        width=tw, guidance_scale=guidance, generator=gen,
                        num_inference_steps=steps, negative_prompt=negative,
                        num_images_per_prompt=1, strength=0.999).images[0]  # fmt: skip
        self.seconds.append(time.time() - t)
        return np.asarray(out.convert("RGB").resize((w, h), Image.LANCZOS))

    def peak_gpu_gb(self) -> float | None:
        if not self.torch.cuda.is_available():
            return None
        return round(self.torch.cuda.max_memory_allocated() / 1e9, 3)


def pipeline_info(inp: Any) -> dict[str, Any]:
    return {"name": getattr(inp, "name", type(inp).__name__)}
