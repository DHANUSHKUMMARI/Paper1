"""CP5: evidence faithfulness (spec section 9.1, CP5 row; decisions of Dhanush, 19 Nov 2026).

Per image (:func:`explain_image`): class probabilities and gate (``predict_tiled``), the
mask-decoder map, and Grad-CAM++ for the edited class (faithfulness on edited images) and for the
predicted class (the M5 regions of a fully generated verdict).

Per edited image (:func:`faithfulness_record`), for each map: EPG and the mask's area fraction,
and on the tile with the largest edited share the deletion / insertion AUC next to the same
curves for a random pixel order.

:func:`randomization` runs the cascading model-randomization test on a batch of tiles (one model
copy), for both maps.

:func:`cp5_verdict`, per map:

- EPG: mean EPG >= 1.5 x mean area fraction AND a one-sided paired Wilcoxon test (EPG > area
  fraction) with p < 0.01;
- randomization: mean SSIM after the last cumulative step (heads, fusion, trace and the LoRA all
  re-initialized) < 0.5;
- deletion: mean deletion AUC below the mean random-order deletion AUC (paired Wilcoxon p
  reported).

Grad-CAM++ is the primary evidence map if it passes all three (``PASS``); otherwise the mask
decoder is primary and Grad-CAM++ secondary (the spec's fallback, ``FALLBACK``); if the decoder
fails as well, ``FAIL`` (stop and ask).
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eviground.data import tiles
from eviground.explain import evidence, faithfulness, gradcam
from eviground.models.heads import CLASSES

EDITED = CLASSES.index("edited")
MAPS = ("gradcam", "decoder")
EPG_RATIO = 1.5
P_MAX = 0.01
SSIM_MAX = 0.5

Amp = Callable[[], contextlib.AbstractContextManager]


def _no_amp() -> contextlib.AbstractContextManager:
    return contextlib.nullcontext()


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float in [0, 1] on ``device``."""
    x = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0  # a copy
    return x.to(device)


def explain_image(
    model: torch.nn.Module, image: np.ndarray, device: torch.device, amp: Amp = _no_amp
) -> dict[str, Any]:
    x = to_tensor(image, device)
    with torch.no_grad(), amp():
        pred = model.predict_tiled(x)
    probs = pred["probs"]
    decided = evidence.verdict(probs, float(model.edit_threshold))
    with amp():
        gc_edited = gradcam.tiled_map(model, x, EDITED)
        gc_pred = (
            gc_edited
            if decided == "edited"
            else gradcam.tiled_map(model, x, CLASSES.index(decided))
        )
    return {
        "probs": probs,
        "gate": pred["gate"],
        "verdict": decided,
        "tiles": pred["tiles"],
        "decoder": pred["mask"].float().cpu().numpy(),
        "gradcam": gc_edited.float().cpu().numpy(),
        "gradcam_pred": gc_pred.float().cpu().numpy(),
    }


def _score_fn(model: torch.nn.Module, amp: Amp) -> Callable[[torch.Tensor], torch.Tensor]:
    def score(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), amp():
            return torch.softmax(model(batch)["logits"].float(), dim=-1)[:, EDITED]

    return score


def faithfulness_record(
    model: torch.nn.Module,
    image: np.ndarray,
    mask: np.ndarray,
    maps: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    deletion: bool = True,
    steps: int = faithfulness.STEPS,
    seed: int = 0,
    amp: Amp = _no_amp,
) -> dict[str, Any]:
    """EPG for every map; with ``deletion``, the curves on the best tile."""
    gt = np.asarray(mask) > 0
    rec: dict[str, Any] = {"area_fraction": faithfulness.area_fraction(gt)}
    for name, m in maps.items():
        rec[f"epg_{name}"] = faithfulness.epg(m, gt)
    if deletion:
        h, w = gt.shape
        left, top, right, bottom = faithfulness.best_tile(gt, tiles.tile_boxes(h, w))
        tile = to_tensor(image, device)[:, top:bottom, left:right]
        score = _score_fn(model, amp)
        rec["tile_edited_share"] = float(gt[top:bottom, left:right].mean())
        for name, m in maps.items():
            cur = faithfulness.deletion_insertion(score, tile, m[top:bottom, left:right], steps)
            rec[f"deletion_{name}"] = cur["deletion_auc"]
            rec[f"insertion_{name}"] = cur["insertion_auc"]
        rnd = faithfulness.random_order_auc(score, tile, (bottom - top, right - left), steps, seed)
        rec["deletion_random"] = rnd["deletion_auc"]
        rec["insertion_random"] = rnd["insertion_auc"]
    return rec


def _norm_each(m: torch.Tensor) -> torch.Tensor:
    top = m.flatten(1).amax(dim=1).clamp_min(1e-12)
    return m / top[:, None, None]


def _batch_maps(
    model: torch.nn.Module, x: torch.Tensor, amp: Amp, batch: int
) -> dict[str, torch.Tensor]:
    gc, dec = [], []
    for i in range(0, x.shape[0], batch):
        part = x[i : i + batch]
        with amp():
            gc.append(gradcam.gradcam_pp(model, part, EDITED).float())
        with torch.no_grad(), amp():
            dec.append(torch.sigmoid(model(part)["mask_logits"].float())[:, 0])
    return {"gradcam": _norm_each(torch.cat(gc)).cpu(), "decoder": _norm_each(torch.cat(dec)).cpu()}


def randomization(
    model: torch.nn.Module,
    tiles_x: torch.Tensor,
    *,
    steps: Sequence[tuple[str, Sequence[str]]] = faithfulness.RANDOMIZATION,
    seed: int = 0,
    amp: Amp = _no_amp,
    batch: int = tiles.TILE_BATCH,
) -> dict[str, list[dict[str, Any]]]:
    """Cascading randomization on one deep copy of ``model``: after each cumulative step, the
    SSIM of every tile's map against the trained model's, for both maps."""
    base = _batch_maps(model, tiles_x, amp, batch)
    work = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in MAPS}
    for step, names in steps:
        for n in names:
            faithfulness._randomize(getattr(work, n), only_lora=(n == "semantic"), gen=gen)
        cur = _batch_maps(work, tiles_x, amp, batch)
        for name in MAPS:
            values = [faithfulness.ssim(base[name][i], cur[name][i]) for i in range(len(tiles_x))]
            out[name].append({"step": step, "ssim_mean": float(np.mean(values)), "ssim": values})
    del work
    return out


def _wilcoxon(a: Sequence[float], b: Sequence[float], alternative: str) -> float:
    from scipy import stats

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 2 or not np.any(d != 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative=alternative).pvalue)


def map_verdict(
    records: Sequence[Mapping[str, Any]], rand_steps: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    epg = [r[f"epg_{name}"] for r in records]
    area = [r["area_fraction"] for r in records]
    mean_epg, mean_area = float(np.mean(epg)), float(np.mean(area))
    ratio = mean_epg / mean_area if mean_area > 0 else float("inf")
    p_epg = _wilcoxon(epg, area, "greater")
    dels = [r for r in records if f"deletion_{name}" in r]
    d_map = [r[f"deletion_{name}"] for r in dels]
    d_rnd = [r["deletion_random"] for r in dels]
    p_del = _wilcoxon(d_map, d_rnd, "less")
    last = rand_steps[-1]["ssim_mean"] if rand_steps else None
    checks = {
        "epg": {
            "mean_epg": mean_epg,
            "mean_area_fraction": mean_area,
            "ratio": ratio,
            "wilcoxon_p": p_epg,
            "images": len(records),
            "pass": ratio >= EPG_RATIO and p_epg < P_MAX,
        },
        "randomization": {
            "steps": [{"step": s["step"], "ssim_mean": s["ssim_mean"]} for s in rand_steps],
            "ssim_last": last,
            "tiles": len(rand_steps[0]["ssim"]) if rand_steps else 0,
            "pass": last is not None and last < SSIM_MAX,
        },
        "deletion": {
            "mean_auc": float(np.mean(d_map)) if d_map else None,
            "mean_random_auc": float(np.mean(d_rnd)) if d_rnd else None,
            "mean_insertion_auc": float(np.mean([r[f"insertion_{name}"] for r in dels]))
            if dels
            else None,
            "mean_random_insertion_auc": float(np.mean([r["insertion_random"] for r in dels]))
            if dels
            else None,
            "wilcoxon_p": p_del,
            "images": len(dels),
            "pass": bool(d_map) and float(np.mean(d_map)) < float(np.mean(d_rnd)),
        },
    }
    checks["pass"] = all(c["pass"] for c in checks.values() if isinstance(c, dict))
    return checks


def cp5_verdict(
    records: Sequence[Mapping[str, Any]], rand: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    per_map = {name: map_verdict(records, rand.get(name, []), name) for name in MAPS}
    if per_map["gradcam"]["pass"]:
        verdict, primary = "PASS", "gradcam"
    elif per_map["decoder"]["pass"]:
        verdict, primary = "FALLBACK", "decoder"
    else:
        verdict, primary = "FAIL", None
    secondary = {"gradcam": "decoder", "decoder": "gradcam"}.get(primary)
    return {
        "verdict": verdict,
        "primary_map": primary,
        "secondary_map": secondary,
        "rule": {"epg_ratio_min": EPG_RATIO, "p_max": P_MAX, "ssim_max": SSIM_MAX},
        "maps": per_map,
    }


def summarize(records: Iterable[Mapping[str, Any]], key: str = "part") -> dict[str, Any]:
    """Mean of every numeric field, per value of ``key`` and over all records."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get(key)), []).append(r)
        groups.setdefault("all", []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        fields = sorted({k for r in rs for k, v in r.items() if isinstance(v, (int, float))})
        out[g] = {"images": len(rs)}
        for f in fields:
            vals = [r[f] for r in rs if isinstance(r.get(f), (int, float))]
            out[g][f] = float(np.mean(vals)) if vals else None
    return out


def overlay_figure(panels: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """One row per image: the image with its region boxes, the true mask (if any), the decoder map
    and Grad-CAM++ (predicted class). ``panels``: dicts with ``image``, ``mask``, ``maps``,
    ``record`` (evidence JSON) and ``title``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = max(1, len(panels))
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for row, p in zip(axes, panels, strict=False):
        row[0].imshow(p["image"])
        for reg in p["record"].get("regions", []):
            x, y, w, h = reg["bbox"]
            row[0].add_patch(Rectangle((x, y), w, h, fill=False, color="yellow", lw=1.5))
            row[0].text(x, y, reg["id"], color="yellow", fontsize=8, va="bottom")
        row[0].set_title(p.get("title", ""), fontsize=8)
        mask = p.get("mask")
        row[1].imshow(np.zeros(p["image"].shape[:2]) if mask is None else mask, cmap="gray")
        row[1].set_title("true edit mask" if mask is not None else "no mask", fontsize=8)
        row[2].imshow(p["maps"]["decoder"], cmap="magma", vmin=0, vmax=1)
        row[2].set_title("mask decoder", fontsize=8)
        row[3].imshow(p["maps"]["gradcam_pred"], cmap="magma", vmin=0, vmax=1)
        row[3].set_title("Grad-CAM++ (predicted class)", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
