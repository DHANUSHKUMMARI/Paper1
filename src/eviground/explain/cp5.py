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
            rho = [spearman(base[name][i], cur[name][i]) for i in range(len(tiles_x))]
            out[name].append(
                {
                    "step": step,
                    "ssim_mean": float(np.mean(values)),
                    "ssim": values,
                    # reported beside SSIM, never instead of it (Dhanush, 24 Nov 2026)
                    "spearman_mean": float(np.nanmean(rho)) if np.isfinite(rho).any() else None,
                    "spearman": rho,
                }
            )
    del work
    return out


def spearman(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Spearman rank correlation of two maps (NaN when one of them is constant)."""
    from scipy import stats

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


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


# --------------------------------------------------------------------------------------
# Token-level deletion (pre-registered repair of the test, Dhanush 24 Nov 2026)
# --------------------------------------------------------------------------------------

TOKEN_GRID = 16
N_RANDOM = 10
TOKEN_ORDERS = ("oracle", "decoder", "gradcam")


def token_scores(map_tile: np.ndarray, grid: int = TOKEN_GRID) -> np.ndarray:
    """A tile's map averaged over each token's cell: (grid * grid,), row-major like the tokens."""
    m = np.asarray(map_tile, dtype=np.float64)
    h, w = m.shape[0] // grid * grid, m.shape[1] // grid * grid
    m = m[:h, :w]
    return m.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3)).ravel()


def _token_curve(
    model: torch.nn.Module,
    s: torch.Tensor,
    t: torch.Tensor,
    order: np.ndarray,
    steps: int,
    amp: Amp,
) -> np.ndarray:
    """P(edited) after removing the first k/steps of ``order``: at the removed grid positions
    both the semantic patch token and the trace token become the tile's mean token; the class
    token is kept. Fusion, gate and class head are re-run; CLIP and the trace CNN are not."""
    n = t.shape[1]
    s_mean = s[:, 1:].mean(dim=1, keepdim=True)
    t_mean = t.mean(dim=1, keepdim=True)
    ss = s.repeat(steps + 1, 1, 1)
    tt = t.repeat(steps + 1, 1, 1)
    for k in range(1, steps + 1):
        gone = torch.as_tensor(order[: round(n * k / steps)], device=s.device, dtype=torch.long)
        ss[k, 1 + gone] = s_mean[0]
        tt[k, gone] = t_mean[0]
    with torch.no_grad(), amp():
        logits = model.logits_from_tokens(ss, tt)
    return torch.softmax(logits.float(), dim=-1)[:, EDITED].cpu().numpy()


def token_deletion(
    model: torch.nn.Module,
    tile: torch.Tensor,
    scores: Mapping[str, np.ndarray],
    *,
    steps: int = faithfulness.STEPS,
    n_random: int = N_RANDOM,
    seed: int = 0,
    amp: Amp = _no_amp,
) -> dict[str, Any]:
    """Token deletion on one tile (3, 256, 256) for each named token ranking in ``scores``
    (highest first) and for ``n_random`` seeded random orders (their mean AUC is the baseline)."""
    with torch.no_grad(), amp():
        s, t = model.branch_tokens(tile[None])
    s, t = s.float(), t.float()
    n = t.shape[1]
    rec: dict[str, Any] = {}
    for name, sc in scores.items():
        order = np.argsort(-np.asarray(sc, dtype=np.float64), kind="stable")
        curve = _token_curve(model, s, t, order, steps, amp)
        rec[f"token_{name}"] = faithfulness._auc(curve)
        rec[f"token_{name}_curve"] = curve.tolist()
    rng = np.random.default_rng(seed)
    rand = [
        faithfulness._auc(_token_curve(model, s, t, rng.permutation(n), steps, amp))
        for _ in range(n_random)
    ]
    rec["token_random"] = float(np.mean(rand))
    rec["token_random_sd"] = float(np.std(rand))
    rec["token_p_full"] = float(curve[0]) if scores else None
    return rec


def token_deletion_verdict(
    records: Sequence[Mapping[str, Any]], maps: Sequence[str] = MAPS
) -> dict[str, Any]:
    """Validity first: the oracle order must beat random (mean AUC lower, one-sided paired
    Wilcoxon p < 0.01). If it does not, the test is uninformative for every map. Otherwise a
    map passes when its AUC is below random with p < 0.01."""
    rnd = [r["token_random"] for r in records]
    orc = [r["token_oracle"] for r in records]
    p_orc = _wilcoxon(orc, rnd, "less")
    valid = bool(records) and float(np.mean(orc)) < float(np.mean(rnd)) and p_orc < P_MAX
    out: dict[str, Any] = {
        "images": len(records),
        "rule": {"p_max": P_MAX, "random_orders": N_RANDOM},
        "validity": {
            "mean_oracle_auc": float(np.mean(orc)) if orc else None,
            "mean_random_auc": float(np.mean(rnd)) if rnd else None,
            "wilcoxon_p": p_orc,
            "valid": valid,
        },
        "maps": {},
    }
    for name in maps:
        vals = [r[f"token_{name}"] for r in records]
        p = _wilcoxon(vals, rnd, "less")
        ok = bool(vals) and float(np.mean(vals)) < float(np.mean(rnd)) and p < P_MAX
        verdict = "UNINFORMATIVE"
        if valid:
            verdict = "PASS" if ok else "FAIL"
        out["maps"][name] = {
            "mean_auc": float(np.mean(vals)) if vals else None,
            "mean_random_auc": float(np.mean(rnd)) if rnd else None,
            "wilcoxon_p": p,
            "verdict": verdict,
        }
    return out


def bundle_tile(
    map_small: np.ndarray, box: Sequence[int], k: int = evidence.DOWNSCALE
) -> np.ndarray:
    """The window ``box`` (left, top, right, bottom; full-resolution pixels) of a bundle map
    saved at 1/k resolution, back at full size (each cell repeated k x k, edge-padded)."""
    left, top, right, bottom = (int(v) for v in box)
    m = np.asarray(map_small, dtype=np.float64)
    sub = m[top // k : -(-bottom // k), left // k : -(-right // k)]
    up = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    up = up[top % k : top % k + (bottom - top), left % k : left % k + (right - left)]
    pad = ((0, bottom - top - up.shape[0]), (0, right - left - up.shape[1]))
    return np.pad(up, pad, mode="edge") if any(p[1] for p in pad) else up


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def aspect_sensitivity(
    p6_records: Sequence[Mapping[str, Any]],
    token_records: Sequence[Mapping[str, Any]],
    excluded: Sequence[str],
    parts: Sequence[str] = ("all", "diffseg_heldout", "sidset_tampered", "brgen"),
) -> dict[str, Any]:
    """CP5 per part as run and without ``excluded`` uids (the aspect-mismatched SID-Set masks;
    a sensitivity check only, Dhanush 24 Nov 2026): EPG ratio per map from notebook 14's P6
    records (CSV rows are fine: numbers may be strings) and the token-deletion verdict from
    notebook 15's records."""
    drop = set(excluded)
    edited = [r for r in p6_records if _num(r.get("epg_gradcam")) is not None]
    out: dict[str, Any] = {"excluded": len(drop), "parts": {}}
    for part in parts:
        row: dict[str, Any] = {}
        for variant, keep in (
            ("as_run", lambda u: True),
            ("excl_mismatched", lambda u: u not in drop),
        ):
            recs = [r for r in edited if keep(r["uid"]) and part in ("all", r.get("part"))]
            area = [_num(r["area_fraction"]) for r in recs]
            epg = {
                name: (
                    float(np.mean([_num(r[f"epg_{name}"]) for r in recs])) / float(np.mean(area))
                )
                if recs and np.mean(area) > 0
                else None
                for name in MAPS
            }
            tok = [r for r in token_records if keep(r["uid"]) and part in ("all", r.get("part"))]
            verdict = token_deletion_verdict(tok) if tok else None
            row[variant] = {
                "edited_images": len(recs),
                "epg_ratio": epg,
                "token_images": len(tok),
                "token_valid": verdict["validity"]["valid"] if verdict else None,
                "token_maps": {k: v["verdict"] for k, v in verdict["maps"].items()}
                if verdict
                else None,
            }
        out["parts"][part] = row
    return out
