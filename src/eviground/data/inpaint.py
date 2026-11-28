"""Inpainting backends for the CP8 Claims-Edit set (Dhanush, 27 Nov 2026).

- :class:`MockInpainter`: deterministic and CPU-only, for tests and the notebook dry run;
- :class:`KolorsInpainter`: Kolors-Inpainting (Hugging Face ``Kwai-Kolors/Kolors-Inpainting``
  @ :data:`KOLORS_WEIGHTS_REV`, Apache-2.0) through the pipeline of the Kolors repository
  (``Kwai-Kolors/Kolors`` @ :data:`KOLORS_REPO_REV`, installed with ``--no-deps``), as the repo's
  ``inpainting/sample_inpainting.py`` builds it, in fp16, loaded in two stages so that it fits a
  T4 with 12.7 GB of system RAM (see the class). Imported lazily: the sandbox and CI never need it.

Every backend maps (image (H, W, 3) uint8, mask (H, W) bool, prompt, seed) to a generated image
of the same shape; :func:`claims_edit.composite` then keeps only the pixels inside the mask.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

KOLORS_WEIGHTS = "Kwai-Kolors/Kolors-Inpainting"
# the full commit of Kwai-Kolors/Kolors-Inpainting (2024-07-26); the earlier value here had only its
# first 12 characters right, so the download would have failed (found 28 Nov, before any run)
KOLORS_WEIGHTS_REV = "3bb98b424c9aa02d11164bf78904a3e7fba17c09"
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


def patch_chatglm_pad(tokenizer_cls: Any) -> bool:
    """Let Kolors' ChatGLM tokenizer run under transformers >= 4.45, which passes ``padding_side``
    to ``_pad``; the tokenizer pads on the left only, so the argument is checked and dropped. The
    token ids are unchanged: every CP8 prompt and the negative prompt gave byte-identical ids and
    masks under transformers 4.42.4 / tokenizers 0.19.1 (Python 3.12) and 4.45.2 / 0.20.3 with
    this patch (sandbox, 28 Nov). Returns whether the patch was applied."""
    import inspect

    if "padding_side" in inspect.signature(tokenizer_cls._pad).parameters:
        return False
    original = tokenizer_cls._pad

    def _pad(self, *args, padding_side=None, **kwargs):
        if padding_side not in (None, "left"):
            raise ValueError(f"the ChatGLM tokenizer pads on the left only, not {padding_side!r}")
        return original(self, *args, **kwargs)

    tokenizer_cls._pad = _pad
    return True


class KolorsInpainter:
    """Kolors-Inpainting, fp16 on the GPU, loaded in two stages (28 Nov).

    A standard Colab T4 has 12.7 GB of system RAM, less than ChatGLM-6B (12.5 GB fp16) plus the
    UNet that model CPU offload keeps in RAM, so the run crashed while loading. Each model fits
    the 15 GB GPU alone, and the text encoder is needed only for the plan's few distinct prompts:

    1. ChatGLM is loaded straight onto the GPU (shard by shard) and every prompt in ``prompts`` is
       encoded with ``negative`` by the pinned pipeline's own ``encode_prompt``; the embeddings are
       kept on the CPU and the encoder is freed;
    2. the UNet and VAE are loaded in fp16 onto the GPU, and the pipeline's ``encode_prompt`` is
       replaced by a lookup into those embeddings.

    Each model runs on the GPU in fp16, as under CPU offload; only where weights wait changes.
    ``weights_dir``: the pinned weights (``snapshot_download``); the pinned repo must be importable.
    """

    name = f"Kolors-Inpainting@{KOLORS_WEIGHTS_REV[:12]} (repo @{KOLORS_REPO_REV[:12]})"

    def __init__(
        self, weights_dir: Path | str, prompts, negative: str, device: str = "cuda"
    ) -> None:
        import gc
        from types import SimpleNamespace

        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
        from kolors.models.modeling_chatglm import ChatGLMModel
        from kolors.models.tokenization_chatglm import ChatGLMTokenizer
        from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import (
            StableDiffusionXLInpaintPipeline,
        )

        patch_chatglm_pad(ChatGLMTokenizer)
        d = Path(weights_dir)
        fp16 = torch.float16
        tokenizer = ChatGLMTokenizer.from_pretrained(d / "text_encoder")
        # stage 1: the text encoder alone on the GPU; the shards go straight there (RAM ~ one shard)
        encoder = ChatGLMModel.from_pretrained(
            d / "text_encoder", torch_dtype=fp16, device_map={"": device}
        ).half()
        stub = SimpleNamespace(
            tokenizer=tokenizer, text_encoder=encoder,
            config=SimpleNamespace(force_zeros_for_empty_prompt=False),
        )  # fmt: skip

        def encode(prompt: str):
            with torch.no_grad():
                out = StableDiffusionXLInpaintPipeline.encode_prompt(
                    stub, prompt=prompt, device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=True, negative_prompt=negative,
                )  # fmt: skip
            return tuple(t.to("cpu") for t in out)

        self.embeds = _encode_all(encode, prompts)
        stub.text_encoder = None  # free the 12.5 GB encoder before the denoiser loads
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        # stage 2: the denoiser on the GPU, in fp16 from the start (no fp32 copy in RAM)
        self.pipe = StableDiffusionXLInpaintPipeline(
            vae=AutoencoderKL.from_pretrained(d / "vae", torch_dtype=fp16),
            text_encoder=None,
            tokenizer=tokenizer,
            unet=UNet2DConditionModel.from_pretrained(d / "unet", torch_dtype=fp16),
            scheduler=EulerDiscreteScheduler.from_pretrained(d / "scheduler"),
        ).to(device)
        self.pipe.enable_attention_slicing()
        self.pipe.encode_prompt = self._lookup  # the pinned encode_prompt needs the encoder
        self.torch = torch
        self.device = device
        self.seconds: list[float] = []

    def _lookup(self, prompt, device=None, **_):
        return tuple(t.to(device or self.device) for t in lookup_embeds(self.embeds, prompt))

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


def _encode_all(encode, prompts) -> dict[str, tuple]:
    """Each distinct prompt encoded once (in first-seen order): {prompt: encode(prompt)}."""
    out: dict[str, tuple] = {}
    for p in prompts:
        if p not in out:
            out[p] = encode(p)
    return out


def lookup_embeds(embeds: dict[str, tuple], prompt: str) -> tuple:
    """The cached embeddings of ``prompt``; a prompt outside the frozen plan is an error."""
    if prompt not in embeds:
        raise KeyError(f"prompt not encoded at load (not in the frozen plan): {prompt!r}")
    return embeds[prompt]


def pipeline_info(inp: Any) -> dict[str, Any]:
    return {"name": getattr(inp, "name", type(inp).__name__)}
