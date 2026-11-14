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

import inspect
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

DEFAULT_BACKBONE = "openai/clip-vit-large-patch14"


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
) -> nn.Module:
    """``revision``: the pinned Hugging Face commit hash of ``backbone`` (config key
    ``model.semantic.revision``); the trainer refuses a real backbone without one."""
    from transformers import CLIPVisionConfig, CLIPVisionModel

    if tiny is not None:
        cfg = CLIPVisionConfig(**{"image_size": 224, "patch_size": 14, **tiny})
        try:
            return CLIPVisionModel._from_config(cfg, attn_implementation="sdpa")
        except (TypeError, ValueError):  # older transformers
            return CLIPVisionModel(cfg)
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
