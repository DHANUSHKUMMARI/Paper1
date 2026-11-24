"""Counterfactual consistency (CC) of evidence regions (spec section 10; rule fixed by Dhanush,
24 Nov 2026, before Phase C).

For one image and one region (a boolean mask at image resolution):

1. The image is cut into its test-time tiles (:func:`eviground.data.tiles.tile_boxes`). Each tile
   runs once through CLIP and the trace CNN (:meth:`EviGround.branch_tokens`).
2. A token belongs to the region when at least ``MIN_TOKEN_SHARE`` (25%) of its pixels lie inside
   it. The region's tokens are removed in **every tile it covers**: at those grid positions the
   semantic patch token and the trace token become the tile's mean token (the class token is
   kept), and fusion, gate and class head are re-run (:meth:`EviGround.logits_from_tokens`), as in
   the CP5 token deletion of notebook 15.
3. The image-level probabilities are re-aggregated with the normal tile rule (mean of P(real) and
   P(generated), maximum of P(edited); :func:`aggregate_tiles`), and the drop of P(predicted
   class) is read, the predicted class being the unperturbed image-level label.
4. The baseline removes an equal-size random token set: in each tile as many tokens as the region
   removed there, drawn uniformly from that tile's tokens (seeded), ``N_RANDOM`` (10) times; its
   drop is the mean over the 10 sets.
5. The region **passes** when its drop is at least ``DELTA`` (0.2) **and** larger than the random
   drop. CC is the share of passing regions.

A region with no token at the 25% share is not scored (``scored`` False) and is counted apart.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from eviground.data import tiles as tiling
from eviground.models.eviground import CLASSES, aggregate_tiles, decide

DELTA = 0.2
MIN_TOKEN_SHARE = 0.25
N_RANDOM = 10
LABEL_TO_CLASS = {"real": "real", "generated": "generated", "edited": "edited"}

Amp = Callable[[], contextlib.AbstractContextManager]


def _no_amp() -> contextlib.AbstractContextManager:
    return contextlib.nullcontext()


def token_membership(
    region_tile: np.ndarray, grid: int = 16, min_share: float = MIN_TOKEN_SHARE
) -> np.ndarray:
    """(grid * grid,) bool, row-major like the tokens: the token's cell is at least
    ``min_share`` inside the region (cells are tile / grid pixels; any tile size)."""
    m = torch.as_tensor(np.asarray(region_tile, dtype=np.float32))[None, None]
    share = F.adaptive_avg_pool2d(m, grid)[0, 0]
    return (share >= min_share - 1e-9).flatten().numpy()


def image_tokens(
    model: torch.nn.Module, image: np.ndarray, amp: Amp = _no_amp, device: str = "cpu"
) -> dict[str, Any]:
    """Branch tokens of every tile of ``image`` (H, W, 3) uint8: ``s`` (n, 1 + N, d), ``t``
    (n, N, d) as float32, the tile ``boxes`` and the token ``grid``. Tiles of different shapes
    (images narrower than a tile) are run one shape at a time."""
    h, w = image.shape[:2]
    boxes = tiling.tile_boxes(h, w)
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255).to(device)
    s_all: list[torch.Tensor | None] = [None] * len(boxes)
    t_all: list[torch.Tensor | None] = [None] * len(boxes)
    by_shape: dict[tuple[int, int], list[int]] = {}
    for i, (le, t, r, b) in enumerate(boxes):
        by_shape.setdefault((b - t, r - le), []).append(i)
    for idx in by_shape.values():
        for j in range(0, len(idx), tiling.TILE_BATCH):
            part = idx[j : j + tiling.TILE_BATCH]
            crops = torch.stack([x[:, t:b, le:r] for le, t, r, b in (boxes[i] for i in part)])
            with torch.no_grad(), amp():
                s, tt = model.branch_tokens(crops)
            for k, i in enumerate(part):
                s_all[i], t_all[i] = s[k].float(), tt[k].float()
    s_stack, t_stack = torch.stack(s_all), torch.stack(t_all)  # type: ignore[arg-type]
    grid = int(round(t_stack.shape[1] ** 0.5))
    return {"s": s_stack, "t": t_stack, "boxes": boxes, "grid": grid}


def _probs(model: torch.nn.Module, s: torch.Tensor, t: torch.Tensor, amp: Amp) -> torch.Tensor:
    with torch.no_grad(), amp():
        logits = model.logits_from_tokens(s, t)
    return torch.softmax(logits.float(), dim=-1)


def remove_tokens(s: torch.Tensor, t: torch.Tensor, sel: torch.Tensor):
    """Copies of ``s`` (n, 1 + N, d) and ``t`` (n, N, d) with the tokens selected by ``sel``
    (n, N) bool replaced by their tile's mean token; the class token is kept."""
    s2, t2 = s.clone(), t.clone()
    s_mean = s[:, 1:].mean(dim=1, keepdim=True).expand_as(s[:, 1:])
    t_mean = t.mean(dim=1, keepdim=True).expand_as(t)
    s2[:, 1:][sel] = s_mean[sel]
    t2[sel] = t_mean[sel]
    return s2, t2


def region_selection(region: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], grid: int):
    """(n, N) bool: the region's tokens in every tile (``MIN_TOKEN_SHARE`` rule)."""
    rows = []
    for le, t, r, b in boxes:
        rows.append(token_membership(region[t:b, le:r], grid))
    return torch.as_tensor(np.stack(rows))


def random_selection(
    counts: Sequence[int], n_tokens: int, rng: np.random.Generator
) -> torch.Tensor:
    """(n, N) bool: in each tile ``counts[i]`` tokens drawn uniformly without replacement."""
    sel = np.zeros((len(counts), n_tokens), dtype=bool)
    for i, k in enumerate(counts):
        if k:
            sel[i, rng.choice(n_tokens, size=int(k), replace=False)] = True
    return torch.as_tensor(sel)


def seed_for(uid: str, region_id: str, base: int = 0) -> int:
    """A stable seed per (image, region), independent of the run order."""
    digest = hashlib.sha256(f"{base}:{uid}:{region_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def region_cc(
    model: torch.nn.Module,
    tokens: Mapping[str, Any],
    region: np.ndarray,
    *,
    seed: int = 0,
    n_random: int = N_RANDOM,
    delta: float = DELTA,
    amp: Amp = _no_amp,
    edit_threshold: float = 0.5,
) -> dict[str, Any]:
    """CC of one region on one image whose tile tokens are ``tokens`` (:func:`image_tokens`)."""
    s, t, boxes, grid = tokens["s"], tokens["t"], tokens["boxes"], tokens["grid"]
    n, n_tok = t.shape[0], t.shape[1]
    base = aggregate_tiles(_probs(model, s, t, amp))
    pred = decide(base, edit_threshold)
    cls = LABEL_TO_CLASS[pred]
    sel = region_selection(region, boxes, grid)
    counts = sel.sum(dim=1).tolist()
    rec: dict[str, Any] = {
        "pred_label": pred,
        "p_pred": base[cls],
        "tokens_removed": int(sum(counts)),
        "tiles_touched": int(sum(1 for c in counts if c)),
        "tiles": n,
    }
    if not rec["tokens_removed"]:
        return {**rec, "scored": False, "drop": None, "random_drop": None, "passed": None}
    rng = np.random.default_rng(seed)
    sels = [sel] + [random_selection(counts, n_tok, rng) for _ in range(n_random)]
    ss, tt = zip(*(remove_tokens(s, t, x.to(s.device)) for x in sels), strict=True)
    p = _probs(model, torch.cat(ss), torch.cat(tt), amp).reshape(len(sels), n, len(CLASSES))
    p_after = [aggregate_tiles(p[k])[cls] for k in range(len(sels))]
    drop = base[cls] - p_after[0]
    rand = [base[cls] - x for x in p_after[1:]]
    random_drop = float(np.mean(rand))
    return {
        **rec,
        "scored": True,
        "p_after": p_after[0],
        "drop": float(drop),
        "random_drop": random_drop,
        "random_drop_sd": float(np.std(rand)),
        "passed": bool(drop >= delta and drop > random_drop),
    }


def cc_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """CC over region records: the share of scored regions that pass, with counts, the mean drops
    and the unscored count."""
    recs = list(records)
    scored = [r for r in recs if r.get("scored")]
    passed = sum(bool(r["passed"]) for r in scored)
    return {
        "regions": len(recs),
        "scored": len(scored),
        "unscored_no_tokens": len(recs) - len(scored),
        "passed": passed,
        "cc": passed / len(scored) if scored else None,
        "mean_drop": float(np.mean([r["drop"] for r in scored])) if scored else None,
        "mean_random_drop": float(np.mean([r["random_drop"] for r in scored])) if scored else None,
        "drop_over_delta": sum(r["drop"] >= DELTA for r in scored),
        "drop_over_random": sum(r["drop"] > r["random_drop"] for r in scored),
    }
