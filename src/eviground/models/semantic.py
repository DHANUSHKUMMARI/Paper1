"""M2 — semantic branch: frozen CLIP ViT-L/14 with LoRA (Research Specification section 8, M2).

``CLIPVisionModel`` from Hugging Face ``transformers`` (``openai/clip-vit-large-patch14``, SDPA
attention), every weight frozen, LoRA (``peft``) on the query and value projections of the last 4
encoder layers. Output: the 257 tokens of the last layer (class token + 16 x 16 patches),
projected to width ``d``.

Efficiency: the layers before the first LoRA layer have no trainable parameter, so they run
under ``torch.no_grad()``; the result is the same tensor, but no activations are kept and the
backward pass stops at the first LoRA layer.

``tiny`` builds a small random CLIP with the same code path, for tests and CI (no download).
"""

from __future__ import annotations

import hashlib
import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_BACKBONE = "openai/clip-vit-large-patch14"
WEIGHTS_FILE = "model.safetensors"
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


class WeightsMismatch(RuntimeError):
    """The CLIP weights file on disk is not the pinned one: stop."""


def weights_sha256(path: Path) -> str:
    """SHA-256 of a weights file (the target, if ``path`` is a link); computed once per file
    version in a process."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(real, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def verified_snapshot(
    backbone: str,
    *,
    revision: str | None,
    sha256: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """The cached snapshot folder of ``backbone`` at ``revision``, after checking that its
    ``model.safetensors`` has the pinned SHA-256 (config key ``model.semantic.weights_sha256``).
    The model is then loaded from this folder, so the file checked is the file loaded."""
    from huggingface_hub import hf_hub_download

    kw = {"revision": revision, "cache_dir": cache_dir, "local_files_only": local_files_only}
    weights = Path(hf_hub_download(backbone, WEIGHTS_FILE, **kw))
    config = Path(hf_hub_download(backbone, "config.json", **kw))
    if config.parent != weights.parent:
        raise WeightsMismatch(f"config.json and {WEIGHTS_FILE} are in different snapshots")
    got = weights_sha256(weights)
    if got != sha256.lower():
        raise WeightsMismatch(
            f"{weights} ({os.path.getsize(os.path.realpath(weights))} bytes) has sha256 "
            f"{got}, not the pinned {sha256}: delete this cache folder and download again (a "
            "Google Drive cache, for example, stores the cache's links as small text files)"
        )
    return weights.parent


def _vision(model: nn.Module) -> nn.Module:
    """transformers 4.x wraps the transformer in ``vision_model``; 5.x does not."""
    return getattr(model, "vision_model", model)


def load_clip(
    backbone: str = DEFAULT_BACKBONE,
    *,
    tiny: dict[str, Any] | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    weights_sha256: str | None = None,
) -> nn.Module:
    """``revision``: the pinned Hugging Face commit hash of ``backbone`` (config key
    ``model.semantic.revision``); ``weights_sha256``: the pinned SHA-256 of its
    ``model.safetensors`` (``model.semantic.weights_sha256``). With a hash, the weights are
    checked and then loaded from the checked snapshot folder. The trainer refuses a real
    backbone without both."""
    from transformers import CLIPVisionConfig, CLIPVisionModel

    if tiny is not None:
        cfg = CLIPVisionConfig(**{"image_size": 224, "patch_size": 14, **tiny})
        try:
            return CLIPVisionModel._from_config(cfg, attn_implementation="sdpa")
        except (TypeError, ValueError):  # older transformers
            return CLIPVisionModel(cfg)
    if weights_sha256:
        snapshot = verified_snapshot(
            backbone,
            revision=revision,
            sha256=weights_sha256,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return CLIPVisionModel.from_pretrained(snapshot, attn_implementation="sdpa")
    return CLIPVisionModel.from_pretrained(
        backbone,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
    )


class SemanticBranch(nn.Module):
    """Frozen CLIP + LoRA on q/v of ``lora_layers``; returns tokens (B, 257, d)."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        *,
        d: int = 512,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_layers: Sequence[int] = (20, 21, 22, 23),
        lora_targets: Sequence[str] = ("q_proj", "v_proj"),
        lora_dropout: float = 0.0,
        tiny: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        weights_sha256: str | None = None,
        clip: nn.Module | None = None,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        model = clip or load_clip(
            backbone,
            tiny=tiny,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            weights_sha256=weights_sha256,
        )
        for p in model.parameters():
            p.requires_grad_(False)
        n_layers = len(_vision(model).encoder.layers)
        layers = sorted({int(i) for i in lora_layers})
        if not layers or layers[0] < 0 or layers[-1] >= n_layers:
            raise ValueError(f"lora_layers {layers} outside 0..{n_layers - 1}")
        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            layers_to_transform=layers,
            layers_pattern="layers",
        )
        self.clip = inject_adapter_in_model(lora, model)
        self.first_trainable = layers[0]
        width = model.config.hidden_size
        self.proj = nn.Linear(width, d)
        self.width = width
        layer = _vision(self.clip).encoder.layers[0]
        self._legacy_layer = "causal_attention_mask" in inspect.signature(layer.forward).parameters

    def _layer(self, layer: nn.Module, h: torch.Tensor) -> torch.Tensor:
        out = layer(h, None, None) if self._legacy_layer else layer(h, None)
        return out[0] if isinstance(out, tuple) else out

    def clip_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """Last-layer tokens of CLIP (B, 257, width); frozen layers without autograd."""
        vt = _vision(self.clip)
        layers = vt.encoder.layers
        with torch.no_grad():
            h = vt.pre_layrnorm(vt.embeddings(rgb))
            for layer in layers[: self.first_trainable]:
                h = self._layer(layer, h)
        for layer in layers[self.first_trainable :]:
            h = self._layer(layer, h)
        return h

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.proj(self.clip_tokens(rgb))

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.clip.named_parameters() if "lora_" in n and p.requires_grad]

    def trainable_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
