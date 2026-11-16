"""CP4 evaluation passes shared by notebooks 08 (localization diagnosis) and 09 (gate).

- :func:`load_trained`: a model from a run's checkpoint, after the frozen-CLIP checks.
- :func:`probe_data`: the probe's training batches (the model's epoch plans, crop rule and
  augmentation, several epochs) and fixed validation crops.
- :func:`joint_predictor`: one pass per tile batch for a model and its CLIP probe (shared trunk).
- :func:`diagnosis_pass` and :func:`p1_pass`: tiled passes that turn whole images into records.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eviground.baselines.clip_probe import ProbeHead
from eviground.data.datasets import ForensicDataset
from eviground.data.transforms import Augment
from eviground.eval import locdiag, thresholds
from eviground.eval.tiled import Item, TiledEvaluator, mask_record
from eviground.models.eviground import EviGround, build_model
from eviground.train import checkpoint as ckpt
from eviground.train.trainer import _collate
from eviground.utils.seed import seed_everything


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_trained(cfg: Any, path: Path, device: torch.device) -> EviGround:
    """``cfg``: the config's ``model`` section; ``path``: a run's ``best.pt``. The model is built
    after seeding with the run's seed, as the trainer does (a tiny random CLIP's weights depend on
    it; the real CLIP's do not), then checked against the checkpoint's frozen-CLIP hash."""
    state = ckpt.load(Path(path))
    seed_everything(int(state.get("seed", 42)))
    model = build_model(cfg)
    sem = dict(cfg.get("semantic", {}))
    revision = "tiny" if sem.get("tiny") else sem.get("revision")
    ckpt.restore_model(
        model, state, clip_revision=revision, clip_sha256=ckpt.frozen_clip_sha256(model)
    )
    return model.to(device).eval()


def probe_data(
    shard_dir: Path,
    *,
    profile: str,
    crop: int,
    crops_per_epoch: int,
    epochs: int,
    seed: int,
    batch: int,
    workers: int,
    augment_cfg: Any = None,
) -> tuple[Callable[[], Iterator[dict]], Callable[[], Iterable[dict]]]:
    """(training batches over ``epochs`` epoch plans, a fresh pass over the validation crops)."""
    n_classes = 2 if profile == "detection" else 3
    train = ForensicDataset(
        shard_dir,
        "train",
        profile=profile,
        crop=crop,
        augment=Augment.from_config(augment_cfg),
        seed=seed,
        n_per_class=crops_per_epoch // n_classes,
    )
    val = ForensicDataset(shard_dir, "val", profile=profile, crop=crop, seed=seed)

    def loader(ds: ForensicDataset) -> DataLoader:
        return DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=_collate)

    def train_batches() -> Iterator[dict]:
        for ep in range(epochs):
            train.set_epoch(ep)
            yield from loader(train)

    return train_batches, lambda: loader(val)


def joint_predictor(
    model: EviGround, head: ProbeHead | None, device: torch.device
) -> Callable[[torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Tile batch -> {"model": ..., "probe": ...} from one pass of the frozen CLIP layers."""

    def predict(x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        with autocast(device):
            out = model(x, probe=head is not None)
        res = {"model": {"logits": out["logits"], "mask_logits": out["mask_logits"]}}
        if head is not None:
            res["probe"] = head(out["probe_cls"], out["probe_patches"], tuple(x.shape[-2:]))
        return res

    return predict


def _marked(prob_map: np.ndarray) -> float:
    return float((prob_map >= 0.5).mean())


def diagnosis_pass(
    predict: Callable,
    items: Iterable[Item],
    device: torch.device,
    *,
    ceilings: bool = True,
    sweep: bool = True,
    progress: Callable[[int], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Edited images -> mask records (and resolution ceilings); real images -> labels and
    marked shares. Every record keeps P(edited) per head; with ``sweep`` edited records also
    keep the boundary counts at :data:`thresholds.BOUNDARY_GRID` (validation threshold sweep).
    Other images are skipped."""
    edited: list[dict[str, Any]] = []
    real: list[dict[str, Any]] = []

    def on_image(item: Item, res: Mapping[str, Mapping[str, Any]]) -> None:
        meta = item.meta
        base = {
            "uid": item.uid,
            "source": meta.get("source"),
            "tiles": res["model"]["tiles"],
            "p_edited": {h: float(r["probs"]["edited"]) for h, r in res.items()},
        }
        if item.mask is not None and item.mask.any():
            edited.append(
                {
                    **base,
                    "share": float(item.mask.mean()),
                    "editors": meta.get("editors") or [],
                    "base_type": meta.get("base_type"),
                    "heads": {
                        h: mask_record(r["prob_map"], item.mask, editors=item.editors)
                        for h, r in res.items()
                    },
                    "labels": {h: r["label"] for h, r in res.items()},
                    "ceilings": locdiag.ceiling_records(item.mask) if ceilings else None,
                    "boundary_grid": {
                        h: thresholds.boundary_grid(r["prob_map"], item.mask)
                        for h, r in res.items()
                    }
                    if sweep
                    else None,
                }
            )
        elif meta.get("label") == "real":
            real.append(
                {
                    **base,
                    "heads": {
                        h: {"label": r["label"], "marked_share": _marked(r["prob_map"])}
                        for h, r in res.items()
                    },
                }
            )
        if progress:
            progress(len(edited) + len(real))

    ev = TiledEvaluator(predict, on_image, device=device, want_maps=True)
    t = time.time()
    for item in items:
        if item.mask is not None or item.meta.get("label") == "real":
            ev.add(item)
    ev.close()
    return {
        "edited": edited,
        "real": real,
        "stats": {
            "images": ev.images,
            "tiles": ev.tiles,
            "max_tiles": ev.max_tiles,
            "seconds": round(time.time() - t, 1),
        },
    }


def p1_pass(
    predict: Callable,
    items: Iterable[Item],
    device: torch.device,
    names: Mapping[str, str],
    version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """P1 images -> one record per head and image (``names`` maps predictor heads to report
    names, e.g. {"model": "detection", "probe": "detection_probe"})."""
    records: list[dict[str, Any]] = []

    def on_image(item: Item, res: Mapping[str, Mapping[str, Any]]) -> None:
        for h, r in res.items():
            records.append(
                {
                    "uid": item.uid,
                    "head": names[h],
                    "version": version,
                    "generator": item.meta.get("generator"),
                    "label": item.meta.get("label"),
                    "p_real": r["probs"]["real"],
                    "p_edited": r["probs"]["edited"],
                    "label_pred": r["label"],
                    "tiles": r["tiles"],
                }
            )

    ev = TiledEvaluator(predict, on_image, device=device, want_maps=False)
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


def p2_summary(records: list[dict[str, Any]], head: str) -> dict[str, Any]:
    """P2 metrics of ``head``: all images, the real-base subset, held-out-editor pixels."""
    from eviground.eval.tiled import per_image, pooled

    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [r["heads"][head] for r in rs]
        if not counts:
            return {"images": 0}
        imgs = [per_image(c) for c in counts]
        f1 = [i["f1"] for i in imgs if i["f1"] is not None]
        bf1 = [i["boundary_f1"] for i in imgs if i["boundary_f1"] is not None]
        acc = [r["labels"][head] == "edited" for r in rs]
        out = {
            "images": len(rs),
            **pooled(counts),
            "mean_f1_per_image": float(np.mean(f1)) if f1 else None,
            "mean_boundary_f1": float(np.mean(bf1)) if bf1 else None,
            "image_acc_edited": float(np.mean(acc)),
        }
        if all("ho_tp" in c for c in counts):
            out["heldout_pixels"] = pooled(counts, prefix="ho_")
        return out

    return {
        "all": block(records),
        "real_base": block([r for r in records if r.get("base_type") == "real"]),
    }
