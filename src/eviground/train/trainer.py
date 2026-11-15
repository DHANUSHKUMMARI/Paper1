"""Trainer (Milestone 3 plan, step 3; Dhanush's fixes of 14 Nov 2026).

- AdamW on the model's two parameter groups (LoRA at ``optim.lr_lora``, the rest at
  ``optim.lr``), weight decay ``optim.weight_decay``; linear warm-up for ``train.warmup_epochs``,
  then cosine to 0; gradient clipping at ``train.grad_clip``; ``train.grad_accum`` micro-batches
  per optimizer step.
- Precision from ``compute.precision``: bf16 autocast, fp16 autocast + GradScaler, or fp32. The
  forward pass runs under autocast; losses and metrics are computed in float32 outside it.
- Epoch ``ep`` trains on :class:`ForensicDataset`'s deterministic plan for (seed, ep); a resumed
  run starts at the next unseen item, and every item's crop and augmentation depend on (seed,
  epoch, item) only, so the batches are the same with any number of loader workers.
- Validation every epoch on one fixed crop per validation image; model selection and early
  stopping (patience ``train.patience``) on the selection score of :func:`metrics.selection_score`.
- Checkpoints (:mod:`eviground.train.checkpoint`): ``step_<n>.pt`` and ``last.pt`` every
  ``checkpoint.every_steps`` optimizer steps and at epoch end, ``best.pt`` on improvement.
- Run folder: ``config.yaml``, ``metrics.csv`` (one row per epoch), ``steps.csv`` (one row per
  optimizer step), ``compute.json`` (GPU hours and compute units, cumulative over resumes).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from eviground.data import pack
from eviground.data.datasets import LABELS, ForensicDataset
from eviground.data.transforms import Augment
from eviground.env import git_commit
from eviground.eval import metrics
from eviground.eval.metrics import PLATEAU_GAIN, plateau_gain
from eviground.models.eviground import build_model, disable_dropout
from eviground.train import checkpoint as ckpt
from eviground.train.losses import eviground_loss
from eviground.utils.seed import make_generator, seed_everything, seed_worker

CLASS_NAMES = {v: k for k, v in LABELS.items()}
LOSS_TERMS = ("total", "ce", "bce", "dice", "att")
LAST = "last.pt"  # the newest checkpoint of a run


class NonFiniteLoss(RuntimeError):
    """A loss term or the gradient norm is NaN or infinite: the run stops (CP3 criterion S1)."""


class KilledForTest(RuntimeError):
    """Raised by ``kill_at_step`` (tests): the run stops right after that optimizer step."""


def data_versions(shard_dir: Path) -> dict[str, Any]:
    """Shard index version (sha256 of ``index.parquet``, manifest sha256) and mask-rule version,
    recorded in every checkpoint and every ``metrics.csv`` row (fix 5)."""
    done = pack.read_done(Path(shard_dir)) or {}
    return {
        "index_sha256": pack.file_sha256(Path(shard_dir) / "index.parquet"),
        "manifest_sha256": done.get("manifest_sha256"),
        "mask_rule": done.get("mask_rule", pack.MASK_RULE),
    }


def _plain(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _autocast(device: torch.device, precision: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "source": [b["source"] for b in batch],
        "uid": [b["uid"] for b in batch],
    }


class Trainer:
    def __init__(
        self,
        cfg: Any,
        shard_dir: Path,
        run_dir: Path,
        *,
        seed: int | None = None,
        profile: str | None = None,
        device: str | torch.device | None = None,
        resume: str | Path | None = "auto",
        log: Callable[[str], None] | None = print,
        kill_at_step: int | None = None,
        progress: bool = False,
    ) -> None:
        self.cfg = _plain(cfg)
        self.seed = int(seed if seed is not None else self.cfg.get("seed", 42))
        tc = dict(self.cfg.get("train", {}))
        self.profile = profile or tc.get("profile", "unified")
        if self.profile == "detection":
            tc.update(tc.get("detection") or {})
        self.tc = tc
        self.say = log or (lambda _m: None)
        self.kill_at_step = kill_at_step
        self.shard_dir, self.run_dir = Path(shard_dir), Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = str(self.cfg.get("compute", {}).get("precision", "fp32"))
        if self.device.type == "cpu":
            self.precision = "fp32"

        seed_everything(self.seed, bool(self.cfg.get("deterministic", True)))
        sem = dict(self.cfg["model"].get("semantic", {}))
        self.clip_revision = "tiny" if sem.get("tiny") else sem.get("revision")
        if not self.clip_revision:
            raise ValueError(
                "model.semantic.revision is not set: pin the Hugging Face commit hash of "
                f"{sem.get('backbone')} before training (fix 2 of 14 Nov 2026)"
            )
        if not sem.get("tiny") and not sem.get("weights_sha256"):
            raise ValueError(
                "model.semantic.weights_sha256 is not set: pin the SHA-256 of "
                f"{sem.get('backbone')}'s model.safetensors before training"
            )
        self.model = build_model(self.cfg["model"]).to(self.device)
        self.clip_sha256 = ckpt.frozen_clip_sha256(self.model)  # once per run (fix 2)
        self.versions = data_versions(self.shard_dir)

        crop = int(self.cfg.get("data", {}).get("crop", 256))
        n_classes = 2 if self.profile == "detection" else 3
        self.batch = int(tc.get("batch_size", 32))
        self.accum = int(tc.get("grad_accum", 1))
        self.train_ds = ForensicDataset(
            self.shard_dir,
            "train",
            profile=self.profile,
            crop=crop,
            augment=Augment.from_config(self.cfg.get("data", {}).get("augment")),
            seed=self.seed,
            n_per_class=int(tc.get("crops_per_epoch", 24000)) // n_classes,
        )
        self.val_ds = ForensicDataset(
            self.shard_dir, "val", profile=self.profile, crop=crop, seed=self.seed
        )
        self.train_eval_ds = self._train_eval_set(int(tc.get("train_eval_images", 2000)), crop)
        self.progress = progress
        self.epochs = int(tc.get("epochs", 20))
        self.micro_per_epoch = math.ceil(len(self.train_ds) / self.batch)
        self.steps_per_epoch = math.ceil(self.micro_per_epoch / self.accum)

        oc = dict(self.cfg.get("optim", {}))
        lr, wd = float(oc.get("lr", 1e-4)), float(oc.get("weight_decay", 0.05))
        self.optimizer = torch.optim.AdamW(  # each group sets its own lr; LoRA's is lr_lora
            self.model.param_groups(lr, float(oc.get("lr_lora", 5e-5)), wd),
            lr=lr,
            weight_decay=wd,
        )
        total = self.epochs * self.steps_per_epoch
        warm = max(1, int(float(tc.get("warmup_epochs", 1)) * self.steps_per_epoch))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: _lr_factor(s, warm, total)
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.precision == "fp16" and self.device.type == "cuda"
        )
        self.state = {
            "epoch": 0,
            "step_in_epoch": 0,  # micro-batches of the current epoch already trained
            "global_step": 0,  # optimizer steps
            "best": None,
            "bad_epochs": 0,
            "stopped": False,
            "gpu_seconds": 0.0,
            # running loss sums of the current epoch (kept in checkpoints, so a resumed epoch
            # reports the mean over all its micro-batches)
            "epoch_loss_sums": dict.fromkeys(LOSS_TERMS, 0.0),
            "epoch_micro": 0,
            "best_epoch": None,
            "selection_history": [],
        }
        OmegaConf.save(OmegaConf.create(self.cfg), self.run_dir / "config.yaml")
        if resume:
            self.resume(resume)

    # ---------------------------------------------------------------- checkpoints

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "model": ckpt.trainable_state(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "run": dict(self.state),
            "rng": ckpt.rng_state(),
            "clip_revision": self.clip_revision,
            "clip_sha256": self.clip_sha256,
            "data": dict(self.versions),
            "profile": self.profile,
            "seed": self.seed,
            "config": self.cfg,
            "git_commit": git_commit(),
        }

    def save(self, name: str) -> Path:
        path = self.run_dir / name
        ckpt.save(path, self.checkpoint_state())
        return path

    def save_step(self) -> None:
        path = self.save(f"step_{self.state['global_step']:07d}.pt")
        ckpt.copy(path, self.run_dir / LAST)
        ckpt.rotate(self.run_dir, int(self.cfg.get("checkpoint", {}).get("keep_last", 2)))

    def resume(self, which: str | Path) -> bool:
        path = self.run_dir / LAST if str(which) == "auto" else Path(which)
        if not path.is_file():
            if str(which) != "auto":
                raise FileNotFoundError(path)
            return False
        state = ckpt.load(path)
        for key, now in (("profile", self.profile), ("seed", self.seed)):
            if state.get(key) != now:
                raise ckpt.CheckpointMismatch(
                    f"the checkpoint was trained with {key} {state.get(key)!r}, this run uses "
                    f"{now!r}: stop"
                )
        ckpt.restore_model(
            self.model, state, clip_revision=self.clip_revision, clip_sha256=self.clip_sha256
        )
        if state.get("data", {}).get("index_sha256") != self.versions["index_sha256"]:
            raise ckpt.CheckpointMismatch(
                "the shard index differs from the one this checkpoint was trained on: stop"
            )
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.state.update(state["run"])
        ckpt.set_rng_state(state["rng"])
        self.say(
            f"resumed {path.name}: epoch {self.state['epoch']}, step in epoch "
            f"{self.state['step_in_epoch']}, global step {self.state['global_step']}"
        )
        return True

    # ---------------------------------------------------------------- loaders

    def loader(self, ds: ForensicDataset, start: int, batch: int, workers: int) -> DataLoader:
        return DataLoader(
            Subset(ds, range(start, len(ds))),
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=seed_worker if workers else None,
            generator=make_generator(self.seed),
            collate_fn=_collate,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,  # each epoch's plan reaches fresh workers
        )

    # ---------------------------------------------------------------- training

    def _loss(self, batch: dict[str, Any], train: bool = True) -> tuple[dict, dict]:
        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["label"].to(self.device)
        m = batch["mask"].to(self.device)
        if self.device.type == "cuda" and self.cfg["model"].get("trace", {}).get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.set_grad_enabled(train), _autocast(self.device, self.precision):
            out = self.model(x)
        lc = dict(self.cfg.get("loss", {}))
        loss = eviground_loss(  # float32, outside autocast (fix 4)
            out,
            y,
            m,
            lambda_seg=float(lc.get("lambda_seg", 1.0)),
            lambda_att=float(lc.get("lambda_att", 0.1)),
            label_smoothing=float(lc.get("label_smoothing", 0.05)),
        )
        return out, loss

    def fit(self, max_steps: int | None = None) -> dict[str, Any]:
        """Train until the last epoch, early stopping or ``max_steps`` optimizer steps."""
        patience = int(self.tc.get("patience", 4))
        while self.state["epoch"] < self.epochs and not self.state["stopped"]:
            ep = self.state["epoch"]
            train_means, reached_max = self._train_epoch(ep, max_steps)
            if reached_max:
                return self.summary()
            self._end_epoch(ep, train_means, self.validate(), patience)
        return self.summary()

    def _train_epoch(self, ep: int, max_steps: int | None) -> tuple[dict[str, float], bool]:
        """The rest of epoch ``ep`` from the saved position; returns the mean loss terms and
        whether ``max_steps`` was reached."""
        workers = int(self.cfg.get("data", {}).get("num_workers", 0))
        self.train_ds.set_epoch(ep)
        self.model.train()
        sums = self.state["epoch_loss_sums"]
        self._t0 = time.time()
        start = self.state["step_in_epoch"] * self.batch
        self.optimizer.zero_grad(set_to_none=True)
        with self._bar(ep) as bar:
            for batch in self.loader(self.train_ds, start, self.batch, workers):
                loss = self._accumulate(ep, batch, sums)
                if loss is None:  # inside an accumulation group
                    continue
                self._step_and_log(ep, loss, batch, bar)
                if self._after_step(max_steps):
                    return {}, True
        self._add_time(self._t0)
        n = max(self.state["epoch_micro"], 1)
        return {k: v / n for k, v in sums.items()}, False

    def _accumulate(self, ep: int, batch: dict, sums: dict[str, float]) -> dict | None:
        """Forward and backward of one micro-batch; its loss terms when an optimizer step follows
        (the last micro-batch of a group or of the epoch), else None."""
        _out, loss = self._loss(batch)
        self._check_finite(ep, loss)
        self.scaler.scale(loss["total"] / self._group_size()).backward()
        for k in LOSS_TERMS:
            sums[k] += float(loss[k].detach())
        self.state["epoch_micro"] += 1
        self.state["step_in_epoch"] += 1
        last_micro = self.state["step_in_epoch"] == self.micro_per_epoch
        if self.state["step_in_epoch"] % self.accum and not last_micro:
            return None
        return loss

    def _step_and_log(self, ep: int, loss: dict, batch: dict, bar: Any) -> None:
        grad_norm = self._optimizer_step(ep)
        self._log_step(ep, loss, batch, grad_norm)
        bar.update(1)
        if self.state["global_step"] % 10 == 0:
            bar.set_postfix(
                loss=f"{float(loss['total'].detach()):.4f}",
                lr=f"{self.optimizer.param_groups[1]['lr']:.2e}",
            )

    def _bar(self, ep: int):
        """A progress bar over the optimizer steps left in epoch ``ep`` (shown in notebooks,
        disabled otherwise)."""
        from tqdm.auto import tqdm

        done = self.state["step_in_epoch"] // self.accum
        return tqdm(
            total=self.steps_per_epoch,
            initial=done,
            desc=f"{self.profile} seed {self.seed} epoch {ep + 1}/{self.epochs}",
            unit="step",
            disable=not self.progress,
        )

    def _check_finite(self, ep: int, loss: dict[str, torch.Tensor]) -> None:
        bad = {
            k: str(float(v.detach()))
            for k, v in loss.items()
            if not torch.isfinite(v.detach()).all()
        }
        if bad:
            self._stop_nonfinite(ep, bad)

    def _stop_nonfinite(self, ep: int, bad: dict[str, float]) -> None:
        """No optimizer step: clear the gradients, keep a checkpoint, record and stop (S1)."""
        self.optimizer.zero_grad(set_to_none=True)
        step = self.state["global_step"] + 1
        record = {"epoch": ep, "global_step": step, "non_finite": bad}
        (self.run_dir / "nonfinite.json").write_text(json.dumps(record, indent=1))
        self.save(LAST)
        raise NonFiniteLoss(f"non-finite values at step {step} (epoch {ep}): {bad}")

    def _group_size(self) -> int:
        """Micro-batches in the accumulation group of the next micro-batch: ``grad_accum``, or
        fewer for the last group of an epoch whose length is not a multiple of it."""
        first = self.state["step_in_epoch"] // self.accum * self.accum
        return min(self.accum, self.micro_per_epoch - first)

    def _clipped_step(self, ep: int | None = None) -> float:
        """Unscale, clip at ``train.grad_clip``, step, clear the gradients; returns the gradient
        norm before clipping.

        With ``ep`` (training), a non-finite norm stops the run **before** the optimizer and the
        scheduler move (S1), so ``last.pt`` keeps the weights of the previous step. Under fp16
        the GradScaler is on and skips such a step itself (an overflow lowers the scale), so the
        run goes on; bf16 and fp32 have no such guard."""
        self.scaler.unscale_(self.optimizer)
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                float(self.tc.get("grad_clip", 1.0)),
            )
        )
        if ep is not None and not math.isfinite(norm) and not self.scaler.is_enabled():
            self._stop_nonfinite(ep, {"grad_norm": str(norm)})
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def _optimizer_step(self, ep: int | None = None) -> float:
        norm = self._clipped_step(ep)
        self.scheduler.step()
        self.state["global_step"] += 1
        return norm

    def _after_step(self, max_steps: int | None) -> bool:
        """Checkpoint every ``checkpoint.every_steps``; the test kill; True at ``max_steps``."""
        step = self.state["global_step"]
        if step % int(self.cfg.get("checkpoint", {}).get("every_steps", 500)) == 0:
            self._add_time(self._t0)
            self._t0 = time.time()
            self.save_step()
        if self.kill_at_step == step:
            raise KilledForTest(f"killed after step {step}")
        if max_steps is not None and step >= max_steps:
            self._add_time(self._t0)
            self.save_step()
            return True
        return False

    def _end_epoch(self, ep: int, train: dict, val: dict, patience: int) -> None:
        score = val.get("selection")
        improved = score is not None and (self.state["best"] is None or score > self.state["best"])
        if improved:
            self.state["best"], self.state["bad_epochs"] = score, 0
            self.state["best_epoch"] = ep
        else:
            self.state["bad_epochs"] += 1
        self.state["selection_history"] = [*self.state["selection_history"], score]
        train_eval = self.evaluate(self.train_eval_ds) if self.train_eval_ds is not None else {}
        self.state["epoch"], self.state["step_in_epoch"] = ep + 1, 0
        self.state["epoch_loss_sums"] = dict.fromkeys(LOSS_TERMS, 0.0)
        self.state["epoch_micro"] = 0
        self.state["stopped"] = self.state["bad_epochs"] >= patience
        self._write_metrics(ep, train, val, train_eval)
        if improved:
            self.save("best.pt")
        self.save_step()
        self._write_compute()
        self.say(
            f"epoch {ep}: loss {train['total']:.4f}, selection {score}, best "
            f"{self.state['best']}, patience {self.state['bad_epochs']}/{patience}"
        )

    # ---------------------------------------------------------------- validation

    @torch.no_grad()
    def predict(self, ds: ForensicDataset, batch: int | None = None) -> dict[str, Any]:
        """Probabilities, binary masks and targets of every item of ``ds`` (eval mode)."""
        self.model.eval()
        bs = int(batch or self.tc.get("val_batch_size", 64))
        thr = float(self.cfg["model"].get("edit_threshold", 0.5))
        probs, preds, targets, labels, sources, uids, gates = [], [], [], [], [], [], []
        for b in self.loader(ds, 0, bs, int(self.cfg.get("data", {}).get("num_workers", 0))):
            out, _ = self._loss(b, train=False)
            probs.append(torch.softmax(out["logits"].float(), -1).cpu().numpy())
            preds.append((torch.sigmoid(out["mask_logits"].float()) >= thr)[:, 0].cpu().numpy())
            targets.append(b["mask"][:, 0].numpy() > 0.5)
            gates.append(out["gate"].float().cpu().numpy()[:, 0])
            labels += b["label"].tolist()
            sources += b["source"]
            uids += b["uid"]
        return {
            "probs": np.concatenate(probs).astype(np.float64),
            "pred_masks": np.concatenate(preds),
            "masks": np.concatenate(targets),
            "labels": np.asarray(labels),
            "sources": sources,
            "uids": uids,
            "gate": np.concatenate(gates),
        }

    def validate(self) -> dict[str, Any]:
        return self.evaluate(self.val_ds)

    def evaluate(self, ds: ForensicDataset) -> dict[str, Any]:
        """Selection score, per-class AUCs, mask metrics, per-source accuracy and the per-source
        mask report (R1) on one fixed crop per image of ``ds``."""
        if len(ds) == 0:
            return {"selection": None}
        p = self.predict(ds)
        ms = metrics.mask_scores(p["pred_masks"], p["masks"], p["labels"])
        sel = metrics.selection_score(self.profile, p["labels"], p["probs"], ms["mask_f1"])
        pred_cls = p["probs"].argmax(1)
        y_pred = [CLASS_NAMES[int(i)] for i in pred_cls]
        y_true = [CLASS_NAMES[int(i)] for i in p["labels"]]
        acc = metrics.per_source_accuracy(y_true, y_pred, p["sources"])
        report = metrics.per_source_mask_report(
            p["pred_masks"], p["labels"], pred_cls, p["sources"]
        )
        return {
            **sel,
            **ms,
            "gate_mean": float(p["gate"].mean()),
            "per_source": acc,
            "per_source_mask": report,
        }

    def _train_eval_set(self, n: int, crop: int) -> ForensicDataset | None:
        """A fixed set of ``n`` training images (classes balanced, seeded), one fixed crop each,
        no augmentation: the training side of the train-validation gap."""
        if n <= 0:
            return None
        rows = _balanced_rows(self.train_ds.rows, n, self.seed + 1)
        return ForensicDataset(
            self.shard_dir,
            "val",
            rows=[{**r, "split": "val"} for r in rows],
            profile=self.profile,
            crop=crop,
            seed=self.seed,
        )

    # ---------------------------------------------------------------- 64-image overfit (CP2)

    def overfit(self) -> dict[str, Any]:
        """CP2 / CP3 (A1): ``train.overfit.images`` training images (classes balanced, seeded),
        one fixed crop each, augmentation and dropout off. Every ``eval_every`` steps it records
        accuracy, loss and the per-image mean mask F1 on the edited images; it runs until accuracy
        has reached ``target`` and mask F1 ``mask_target`` (when there are edited images), or
        ``mask_steps`` steps pass. Accuracy must be reached within ``steps``."""
        oc = dict(self.tc.get("overfit") or {})
        n, steps = int(oc.get("images", 64)), int(oc.get("steps", 300))
        mask_steps = max(steps, int(oc.get("mask_steps", 1000)))
        target, every = float(oc.get("target", 0.99)), int(oc.get("eval_every", 25))
        mask_target = float(oc.get("mask_target", 0.80))
        rows = _balanced_rows(self.train_ds.rows, n, self.seed)
        fixed = self._fixed_crops(rows)
        has_edited = any(r["label"] == "edited" for r in rows)
        disable_dropout(self.model)
        for g in self.optimizer.param_groups:  # the configured rates, no warm-up or decay
            g["lr"] = g["initial_lr"]
        batches = _cycle(lambda: self.loader(fixed, 0, self.batch, 0))
        history: list[dict[str, Any]] = []
        acc_at = mask_at = None
        for step in range(1, mask_steps + 1):
            loss = self._overfit_step(next(batches))
            if step % every and step != mask_steps:
                continue
            point = self._overfit_point(fixed, step, loss)
            history.append(point)
            acc_at = acc_at or (step if point["acc"] >= target else None)
            mask_ok = has_edited and (point["mask_f1"] or 0.0) >= mask_target
            mask_at = mask_at or (step if mask_ok else None)
            if acc_at and (mask_at or not has_edited):
                break
        return _overfit_result(
            rows,
            history,
            {"acc": target, "mask_f1": mask_target, "acc_steps": steps, "mask_steps": mask_steps},
            acc_at,
            mask_at if has_edited else None,
        )

    def _fixed_crops(self, rows: list[dict[str, Any]]) -> ForensicDataset:
        """One fixed crop per row (seeded by uid), no augmentation."""
        fixed = ForensicDataset(
            self.shard_dir,
            "val",
            rows=[{**r, "split": "val"} for r in rows],
            crop=self.train_ds.crop,
            seed=self.seed,
        )
        assert not fixed.augment.enabled
        return fixed

    def _overfit_step(self, batch: dict[str, Any]) -> float:
        self.model.train()
        _out, loss = self._loss(batch)
        self.scaler.scale(loss["total"]).backward()
        self._clipped_step()
        return float(loss["total"].detach())

    def _overfit_point(self, ds: ForensicDataset, step: int, loss: float) -> dict[str, Any]:
        p = self.predict(ds, self.batch)
        ms = metrics.mask_scores(p["pred_masks"], p["masks"], p["labels"])
        acc = float((p["probs"].argmax(1) == p["labels"]).mean())
        return {"step": step, "loss": loss, "acc": acc, "mask_f1": ms["mask_f1"]}

    # ---------------------------------------------------------------- logs

    def _log_step(self, ep: int, loss: dict, batch: dict, grad_norm: float) -> None:
        row = {
            "global_step": self.state["global_step"],
            "epoch": ep,
            "step_in_epoch": self.state["step_in_epoch"],
            **{f"loss_{k}": round(float(loss[k].detach()), 6) for k in LOSS_TERMS},
            "grad_norm": round(grad_norm, 6),
            "lr": self.optimizer.param_groups[1]["lr"],
            "lr_lora": self.optimizer.param_groups[0]["lr"],
            "first_uid": batch["uid"][0],
        }
        _append_csv(self.run_dir / "steps.csv", row)

    def _write_metrics(self, ep: int, train: dict, val: dict, train_eval: dict) -> None:
        row: dict[str, Any] = {
            "epoch": ep,
            "global_step": self.state["global_step"],
            "lr": self.optimizer.param_groups[1]["lr"],
            "lr_lora": self.optimizer.param_groups[0]["lr"],
            **{f"train_{k}": round(v, 6) for k, v in train.items()},
        }
        for prefix, res in (("val", val), ("train_eval", train_eval)):
            for k, v in res.items():
                if k == "per_source_mask":
                    row.update({f"{prefix}_{kk}": round(vv, 6) for kk, vv in v.items()})
                elif k != "per_source":
                    row[f"{prefix}_{k}"] = None if v is None else round(float(v), 6)
        row["best"] = self.state["best"]
        row["best_epoch"] = self.state["best_epoch"]
        row["bad_epochs"] = self.state["bad_epochs"]
        if "per_source" in val:
            row.update(metrics.metrics_row(ep, val["per_source"]))
        row.update(self.versions)  # fix 5
        row["clip_revision"] = self.clip_revision
        _append_csv(self.run_dir / "metrics.csv", row)

    def _add_time(self, t0: float) -> None:
        if self.device.type == "cuda":
            self.state["gpu_seconds"] += time.time() - t0

    def _write_compute(self) -> None:
        rate = float(self.cfg.get("compute", {}).get("cu_per_hour", 0.0))
        hours = self.state["gpu_seconds"] / 3600
        info = {
            "gpu_hours": round(hours, 4),
            "cu_per_hour": rate,
            "compute_units": round(hours * rate, 3),
            "global_step": self.state["global_step"],
            "epoch": self.state["epoch"],
        }
        (self.run_dir / "compute.json").write_text(json.dumps(info, indent=1))

    def summary(self) -> dict[str, Any]:
        gain = plateau_gain(self.state["selection_history"])
        still = (
            not self.state["stopped"]
            and self.state["epoch"] >= self.epochs
            and (gain is None or gain >= PLATEAU_GAIN)
        )
        return {
            **{k: self.state[k] for k in ("epoch", "global_step", "best", "stopped")},
            "best_epoch": self.state["best_epoch"],
            "plateau_gain": gain,
            "still_improving_at_end": still,
            "run_dir": str(self.run_dir),
            "clip_revision": self.clip_revision,
            "clip_sha256": self.clip_sha256,
            **self.versions,
        }


def _cycle(make: Callable[[], Any]):
    """Batches of ``make()`` again and again."""
    while True:
        yield from make()


def _lr_factor(step: int, warm: int, total: int) -> float:
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    new = not path.is_file()
    if new:
        header = list(row)
    else:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh))
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _balanced_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` rows with the classes as balanced as the pools allow, chosen by a seeded hash."""

    def key(r: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}/{r['uid']}".encode()).hexdigest()

    by_label: dict[str, list[dict[str, Any]]] = {}
    for r in sorted(rows, key=key):
        by_label.setdefault(r["label"], []).append(r)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(by_label.values()):
        for label in sorted(by_label):
            if by_label[label] and len(picked) < n:
                picked.append(by_label[label].pop(0))
    return picked


def _overfit_result(
    rows: list, history: list, targets: dict, acc_at: int | None, mask_at: int | None
) -> dict:
    """The A1 record: when accuracy and mask F1 reached their targets, and the trajectory."""
    edited = sum(1 for r in rows if r["label"] == "edited")
    acc_pass = acc_at is not None and acc_at <= targets["acc_steps"]
    mask_pass = None if edited == 0 else mask_at is not None and mask_at <= targets["mask_steps"]
    return {
        "images": len(rows),
        "by_class": {c: sum(1 for r in rows if r["label"] == c) for c in LABELS},
        "targets": targets,
        "acc_reached_at_step": acc_at,
        "mask_f1_reached_at_step": mask_at,
        "reached_at_step": acc_at if acc_pass else None,  # CP2's field (accuracy)
        "acc_pass": acc_pass,
        "mask_pass": mask_pass,
        "final_acc": history[-1]["acc"] if history else None,
        "final_mask_f1": history[-1]["mask_f1"] if history else None,
        "history": history,
    }
