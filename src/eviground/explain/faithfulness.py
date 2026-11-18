"""Explanation faithfulness (spec section 10; CP5).

- :func:`epg`: energy pointing game, the share of attribution energy inside the true edit.
- :func:`remove_region`: replaces a region with a blurred copy of the image; the one function used
  by deletion/insertion and by counterfactual consistency (Dhanush, 18 Nov 2026).
- :func:`deletion_insertion`: RISE protocol, the AUC of the edited-class probability while pixels
  are removed (or restored) in attribution order; run on the single tile with the largest edited
  share (:func:`best_tile`).
- :func:`ssim` and :func:`randomization_check`: the model-randomization sanity test (Adebayo et
  al.): layers are re-initialized from the top down and the map must change (SSIM < 0.5).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

STEPS = 20
BLUR_KERNEL = 31
RANDOMIZATION = (
    ("heads", ("cls_head", "decoder")),
    ("fusion", ("fusion",)),
    ("trace", ("trace",)),
    ("semantic_lora", ("semantic",)),
)


def epg(attr: np.ndarray, gt: np.ndarray) -> float:
    a = np.clip(np.asarray(attr, dtype=np.float64), 0, None)
    total = a.sum()
    return float((a * (np.asarray(gt) > 0)).sum() / total) if total > 0 else 0.0


def area_fraction(gt: np.ndarray) -> float:
    return float((np.asarray(gt) > 0).mean())


def blurred(image: torch.Tensor, kernel: int = BLUR_KERNEL) -> torch.Tensor:
    """A strong box blur of (3, H, W) (reflect padding), the removal baseline."""
    pad = kernel // 2
    x = F.pad(image[None], (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel, stride=1)[0]


def remove_region(
    image: torch.Tensor, region: torch.Tensor | np.ndarray, base: torch.Tensor | None = None
) -> torch.Tensor:
    """``image`` (3, H, W) with the pixels of ``region`` (H, W, bool) replaced by the blurred
    image (or by ``base``)."""
    m = torch.as_tensor(np.asarray(region, dtype=bool), device=image.device)
    base = blurred(image) if base is None else base
    return torch.where(m[None], base, image)


def _orders(attr: np.ndarray, steps: int) -> list[np.ndarray]:
    order = np.argsort(-np.asarray(attr, dtype=np.float64).ravel(), kind="stable")
    n = order.size
    masks = []
    for k in range(steps + 1):
        m = np.zeros(n, dtype=bool)
        m[order[: round(n * k / steps)]] = True
        masks.append(m.reshape(np.asarray(attr).shape))
    return masks


def _auc(ys: Sequence[float]) -> float:
    ys = np.asarray(ys, dtype=np.float64)
    return float(((ys[1:] + ys[:-1]) / 2).sum() / (len(ys) - 1))


def deletion_insertion(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    attr: np.ndarray,
    steps: int = STEPS,
) -> dict[str, Any]:
    """Deletion: remove the top-k% attributed pixels; insertion: start from the removed image and
    restore them. ``score_fn`` maps (N, 3, H, W) to the edited-class probability (N,)."""
    base = blurred(image)
    masks = _orders(attr, steps)
    deleted = torch.stack([remove_region(image, m, base) for m in masks])
    inserted = torch.stack([remove_region(image, ~m, base) for m in masks])
    with torch.no_grad():
        d = score_fn(deleted).float().cpu().numpy()
        i = score_fn(inserted).float().cpu().numpy()
    return {
        "deletion_auc": _auc(d),
        "insertion_auc": _auc(i),
        "deletion": d.tolist(),
        "insertion": i.tolist(),
    }


def random_order_auc(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    shape: tuple[int, int],
    steps: int = STEPS,
    seed: int = 0,
) -> dict[str, Any]:
    """The same curves with a random pixel order (the baseline deletion must beat)."""
    rand = np.random.default_rng(seed).random(shape)
    return deletion_insertion(score_fn, image, rand, steps)


def best_tile(mask: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, ...]:
    """The tile box (left, top, right, bottom) with the largest edited share."""
    m = np.asarray(mask) > 0
    return max(boxes, key=lambda b: m[b[1] : b[3], b[0] : b[2]].mean())


def _gauss(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Mean SSIM of two maps in [0, 1] (Gaussian window 11, sigma 1.5)."""
    x = torch.as_tensor(np.asarray(a), dtype=torch.float64)[None, None]
    y = torch.as_tensor(np.asarray(b), dtype=torch.float64)[None, None]
    w = _gauss()
    pad = w.shape[-1] // 2
    mu_x, mu_y = (F.conv2d(F.pad(t, (pad,) * 4, mode="reflect"), w) for t in (x, y))
    sxx = F.conv2d(F.pad(x * x, (pad,) * 4, mode="reflect"), w) - mu_x**2
    syy = F.conv2d(F.pad(y * y, (pad,) * 4, mode="reflect"), w) - mu_y**2
    sxy = F.conv2d(F.pad(x * y, (pad,) * 4, mode="reflect"), w) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def _randomize(module: torch.nn.Module, only_lora: bool, gen: torch.Generator) -> None:
    """Re-initialize ``module``: every sub-layer with ``reset_parameters`` (Linear, Conv,
    LayerNorm, BatchNorm ...) is reset (a fresh random init, norms back to 1/0); with
    ``only_lora``, only the LoRA matrices get small random values (the frozen CLIP stays)."""
    if only_lora:
        with torch.no_grad():
            for name, p in module.named_parameters():
                if "lora" in name:
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
        return
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    for sub in module.modules():
        reset = getattr(sub, "reset_parameters", None)
        if callable(reset):
            reset()


def randomization_check(
    model: torch.nn.Module,
    map_fn: Callable[[torch.nn.Module], torch.Tensor],
    steps: Sequence[tuple[str, Sequence[str]]] = RANDOMIZATION,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Cascading randomization on a deep copy: after each step (cumulative), the SSIM between
    the map of the trained model and the map of the randomized one (both normalized by their
    maximum). ``map_fn(model)`` returns the map."""

    def norm(m):
        m = torch.as_tensor(m).float()
        return m / m.max() if m.max() > 0 else m

    base = norm(map_fn(model)).cpu()
    work = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    out = []
    for step, names in steps:
        for n in names:
            _randomize(getattr(work, n), only_lora=(n == "semantic"), gen=gen)
        out.append({"step": step, "ssim": ssim(base, norm(map_fn(work)).cpu())})
    return out
