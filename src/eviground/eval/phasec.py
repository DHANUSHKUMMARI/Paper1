"""Phase C, notebook 17: the full test-set evaluation in one L4 session (Dhanush, 24 Nov 2026).

- :func:`run_block`: blocks run in priority order. Before each block, if the CU used so far (L4
  rate x wall time since the session start) plus the block's estimate exceeds ``CU_CAP``, the
  session stops cleanly: what is done is saved, the block and every later one are logged as
  skipped, and the notebook goes straight to the compute record and the release. A block whose
  result file is already on Drive is reused (a dropped session restarts where it stopped).
- :func:`items_from_rows`: whole images from staged test-set rows (test-set tars, notebook 16):
  DiffSeg30k masks as in training (AI base read as 0), BR-Gen and SID-Set masks at L > 127.
- :func:`score`: every image through one predictor (tiled, full coverage) into one record per head
  and image: probabilities, label, tiles, and the pixel counts when the image has a mask.
- Summaries per protocol: :func:`pixel_summary` (pooled edited-class IoU, F1, AUC, boundary F1),
  :func:`p3_summary` (BR-Gen per method, region, source and edited-share bin),
  :func:`p3b_summary` (3-class accuracy, per-class F1, pixels on tampered images),
  :func:`p4_summary` (Chameleon accuracy, real and fake recall), :func:`p5_summary` (per
  perturbation condition).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data import diffseg_mask
from eviground.eval import perturb
from eviground.eval.cp4 import average_precision
from eviground.eval.tiled import Item, TiledEvaluator, mask_record, per_image, pooled
from eviground.models.heads import CLASSES

L4_RATE = 2.2
SHARE_BINS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.01))
BIN_NAMES = ("<1%", "1-5%", "5-15%", "15-40%", ">40%")
BINARY_SOURCES = ("brgen", "sidset", "sidset_val")  # masks binarized at L > 127


# ---- blocks under a CU cap -------------------------------------------------------------------


def cu_used(t0: float, rate: float = L4_RATE, now: float | None = None) -> float:
    return ((time.time() if now is None else now) - t0) / 3600 * rate


def run_block(
    name: str,
    est_cu: float,
    fn: Callable[[], Mapping[str, Any]],
    *,
    out_dir: Path,
    log: dict[str, Any],
    t0: float,
    cu_cap: float,
    rate: float = L4_RATE,
    clock: Callable[[], float] = time.time,
) -> Mapping[str, Any] | None:
    """Run one block, or reuse its saved result, or skip it (and stop the session) when the cap
    would be passed. ``log`` collects a status per block; once one block is skipped for the cap,
    ``log['stopped']`` is set and every later block is skipped too."""
    path = Path(out_dir) / f"block_{name}.json"
    if path.is_file():
        log.setdefault("blocks", {})[name] = {"status": "reused"}
        return json.loads(path.read_text())
    used = cu_used(t0, rate, clock())
    if log.get("stopped") or used + est_cu > cu_cap:
        log.setdefault("blocks", {})[name] = {
            "status": "skipped",
            "reason": "an earlier block hit the cap"
            if log.get("stopped")
            else f"CU used {used:.2f} + estimate {est_cu:.2f} > cap {cu_cap:.2f}",
        }
        log["stopped"] = True
        return None
    start = clock()
    result = dict(fn())
    minutes = round((clock() - start) / 60, 2)
    result["_block"] = {"name": name, "minutes": minutes, "est_cu": est_cu, "cu_before": used}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=_json_default))
    log.setdefault("blocks", {})[name] = {"status": "done", "minutes": minutes}
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---- images from staged rows -----------------------------------------------------------------


def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _read_mask(r: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not r.get("mask_path"):
        return None, None
    with Image.open(r["mask_path"]) as m:
        raw = np.asarray(m.convert("L"))
    if str(r.get("source", "")) in BINARY_SOURCES:
        return raw > 127, None
    base = r.get("mask_base")
    base = None if base in (None, "") else int(float(base))
    mask = diffseg_mask.edit_mask(raw, base) > 0
    return mask, np.where(mask, raw, 0)


def items_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: str | None = None,
    level: Any = None,
    keep: Sequence[str] = ("source", "label", "generator", "base_type", "role", "split"),
) -> Iterator[Item]:
    """Whole images of staged rows, optionally perturbed (P5); a mask is resized with the image."""
    for r in rows:
        img = _read_rgb(r["path"])
        mask, editors = _read_mask(r)
        img, mask2 = perturb.apply(img, kind, level, uid=str(r["uid"]), mask=mask)
        if mask2 is not None and mask is not None and mask2.shape != mask.shape:
            editors = None  # per-pixel editors are not resampled
        meta = {k: r.get(k) for k in keep}
        meta["condition"] = perturb.condition_name(kind, level)
        yield Item(str(r["uid"]), img, mask2, editors, meta)


# ---- scoring ---------------------------------------------------------------------------------


def score(
    predict: Callable,
    items: Iterable[Item],
    device: Any,
    *,
    heads: Mapping[str, str] | None = None,
    masks: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One record per (head, image): ``uid``, ``head`` (renamed by ``heads``), the meta fields,
    ``probs``, ``label_pred``, ``tiles`` and, when the image has a mask and ``masks``, the pixel
    counts (:func:`mask_record`)."""
    records: list[dict[str, Any]] = []

    def on_image(item: Item, res: Mapping[str, Mapping[str, Any]]) -> None:
        for h, r in res.items():
            rec = {
                "uid": item.uid,
                "head": (heads or {}).get(h, h),
                **item.meta,
                "probs": dict(r["probs"]),
                "label_pred": r["label"],
                "tiles": r["tiles"],
            }
            if masks and item.mask is not None:
                rec["mask"] = mask_record(r["prob_map"], item.mask, editors=item.editors)
            records.append(rec)

    ev = TiledEvaluator(predict, on_image, device=device, want_maps=masks)
    t = time.time()
    for item in items:
        ev.add(item)
    ev.close()
    stats = {
        "images": ev.images,
        "tiles": ev.tiles,
        "max_tiles": ev.max_tiles,
        "seconds": round(time.time() - t, 1),
    }
    return records, stats


def strip_hist(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Records for a CSV / JSON line: the pixel histograms dropped, the per-image metrics kept."""
    out = []
    for r in records:
        rec = {k: v for k, v in r.items() if k != "mask"}
        if "mask" in r:
            rec.update({f"px_{k}": v for k, v in per_image(r["mask"]).items()})
        out.append(rec)
    return out


# ---- summaries -------------------------------------------------------------------------------


def _mean(v: Sequence[float]) -> float | None:
    return float(np.mean(v)) if len(v) else None


def pixel_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled pixel metrics (edited-class IoU = the DiffSeg30k mIoU, F1, AUC), mean per-image F1 and
    boundary F1, and the image-level edited accuracy, over records with pixel counts."""
    rs = [r for r in records if "mask" in r]
    if not rs:
        return {"images": 0}
    counts = [r["mask"] for r in rs]
    imgs = [per_image(c) for c in counts]
    out = {
        "images": len(rs),
        **pooled(counts),
        "mean_f1_per_image": _mean([i["f1"] for i in imgs if i["f1"] is not None]),
        "mean_boundary_f1": _mean([i["boundary_f1"] for i in imgs if i["boundary_f1"] is not None]),
        "image_acc_edited": _mean([r["label_pred"] == "edited" for r in rs]),
    }
    if all("ho_tp" in c for c in counts):
        out["heldout_pixels"] = pooled(counts, prefix="ho_")
    return out


def edited_share_bin(mask_counts: Mapping[str, Any]) -> str:
    share = mask_counts["gt_pixels"] / max(1, mask_counts["pixels"])
    for (lo, hi), name in zip(SHARE_BINS, BIN_NAMES, strict=True):
        if lo <= share < hi:
            return name
    return BIN_NAMES[-1]


def brgen_parts(uid: str) -> dict[str, str]:
    """Method, region and source of a BR-Gen forged file (``BR-Gen/Forged/<M>/<R>/<S>/...``)."""
    p = uid.split("/")
    i = p.index("Forged") if "Forged" in p else -1
    if i < 0 or len(p) < i + 4:
        return {"method": "?", "region": "?", "source": "?"}
    return {"method": p[i + 1], "region": p[i + 2], "source": p[i + 3]}


def _grouped(records: Sequence[Mapping[str, Any]], key: Callable) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return {k: pixel_summary(v) for k, v in sorted(groups.items())}


def p3_summary(records: Sequence[Mapping[str, Any]], seen_editor: str = "SDXL") -> dict[str, Any]:
    """BR-Gen pixel metrics: all, the four unseen methods, SDXL apart (a training editor), per
    method, region, source and edited-share bin, and ImageNet + Places alone."""
    parts = {r["uid"]: brgen_parts(r["uid"]) for r in records}
    unseen = [r for r in records if parts[r["uid"]]["method"] != seen_editor]
    return {
        "all": pixel_summary(records),
        "unseen_methods": pixel_summary(unseen),
        "sdxl": pixel_summary([r for r in records if parts[r["uid"]]["method"] == seen_editor]),
        "imagenet_places_unseen": pixel_summary(
            [r for r in unseen if parts[r["uid"]]["source"] in ("ImageNet", "Places")]
        ),
        "by_method": _grouped(records, lambda r: parts[r["uid"]]["method"]),
        "by_region": _grouped(unseen, lambda r: parts[r["uid"]]["region"]),
        "by_source": _grouped(unseen, lambda r: parts[r["uid"]]["source"]),
        "by_edited_share": _grouped(unseen, lambda r: edited_share_bin(r["mask"])),
    }


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def p3b_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """SID-Set validation (P3b): 3-class accuracy, per-class F1 (tampered = edited), the confusion,
    and the pixel metrics on tampered images (the class was never trained on)."""
    truth = {"real": "real", "full_synthetic": "generated", "generated": "generated",
             "tampered": "edited", "edited": "edited"}  # fmt: skip
    pairs = [(truth.get(str(r["label"]), str(r["label"])), r["label_pred"]) for r in records]
    conf = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for t, p in pairs:
        if t in conf:
            conf[t][p] += 1
    f1 = {}
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in CLASSES if t != c)
        fn = sum(conf[c][p] for p in CLASSES if p != c)
        f1[c] = _f1(tp, fp, fn)
    return {
        "images": len(records),
        "acc": _mean([t == p for t, p in pairs]),
        "f1": f1,
        "macro_f1": _mean([v for v in f1.values() if v is not None]),
        "confusion": conf,
        "tampered_pixels": pixel_summary([r for r in records if "mask" in r]),
    }


def p4_summary(records: Sequence[Mapping[str, Any]], head: str) -> dict[str, Any]:
    """Chameleon (P4): fake = not real. Accuracy, real recall, fake recall, AP of 1 - P(real)."""
    rs = [r for r in records if r["head"] == head]
    fake = [r["label"] != "real" for r in rs]
    called_fake = [r["probs"]["real"] < 0.5 for r in rs]
    return {
        "images": len(rs),
        "acc": _mean([a == b for a, b in zip(fake, called_fake, strict=True)]),
        "real_recall": _mean([not c for f, c in zip(fake, called_fake, strict=True) if not f]),
        "fake_recall": _mean([c for f, c in zip(fake, called_fake, strict=True) if f]),
        "ap": average_precision(fake, [1 - r["probs"]["real"] for r in rs]),
    }


def p5_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per head and condition: GenImage records -> accuracy (1 - P(real) >= 0.5 is generated) and
    AP; DiffSeg30k records -> image edited accuracy and pooled pixel metrics."""
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        groups[(r["head"], r["condition"])].append(r)
    for (head, cond), rs in sorted(groups.items()):
        gen = [r for r in rs if r.get("source") == "genimage"]
        dif = [r for r in rs if r.get("source") == "diffseg30k"]
        row: dict[str, Any] = {}
        if gen:
            fake = [r["label"] == "generated" for r in gen]
            sc = [1 - r["probs"]["real"] for r in gen]
            row["genimage"] = {
                "images": len(gen),
                "acc": _mean([(s >= 0.5) == f for s, f in zip(sc, fake, strict=True)]),
                "ap": average_precision(fake, sc),
            }
        if dif:
            row["diffseg30k"] = pixel_summary(dif)
        out.setdefault(head, {})[cond] = row
    return out


def param_counts(model: Any) -> dict[str, int]:
    params = list(model.parameters())
    return {
        "total": int(sum(p.numel() for p in params)),
        "trainable": int(sum(p.numel() for p in params if p.requires_grad)),
    }
