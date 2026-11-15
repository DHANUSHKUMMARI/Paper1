"""Training and validation datasets over the packed shards (Milestone 3 plan, step 2).

The shards (:mod:`eviground.data.pack`) hold the harmonized bytes and the 1-bit edit masks, so
loading is: seek to the stored offset, decode, crop, augment. Nothing is harmonized at load time
and nothing is cached on local disk.

Training (``split="train"``): every epoch has a deterministic plan from (seed, epoch)
(:func:`epoch.epoch_plan`): the sampler's class and source shares, and for each draw a resize
factor and a crop box. An edited image's crop follows :func:`crops.edited_crop` on its (resized)
edit mask; a failed draw is replaced from the same pool. Edited images that can never give a
valid crop (``valid_crop`` False in the index) are left out and listed in ``dropped``. Item ``k``
of an epoch is the same in every process and after a restart, so a resumed run skips the draws
already seen by starting at ``k``.

Validation (``split="val"``): one fixed crop per image, drawn from (seed, uid), no augmentation;
an edited image gets an edited crop (the CP1 rule, then, if its tries fail, a window chosen
among all valid windows).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from eviground.data import crops, epoch, pack
from eviground.data.transforms import Augment, resize_image, resize_mask, scaled_size

log = logging.getLogger(__name__)

LABELS = {"real": 0, "generated": 1, "edited": 2}
Spec = tuple[int, int, int, int, float]  # left, top, right, bottom (resized image), scale


def _uid_seed(seed: int, uid: str) -> list[int]:
    return [seed, int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little")]


class ShardReader:
    """Reads images and edit masks from the shards by offset.

    Each shard is opened once per process and read with ``os.pread``, which takes the offset as
    an argument: DataLoader workers forked from one process can share the descriptors safely.
    Descriptors are not pickled; a spawned worker opens its own.
    """

    def __init__(self, shard_dir: Path, rows: Sequence[dict[str, Any]] | None = None) -> None:
        self.shard_dir = Path(shard_dir)
        self.rows = (
            list(rows) if rows is not None else pack.read_index(self.shard_dir / pack.INDEX_FILE)
        )
        self._fds: dict[str, int] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_fds"] = {}
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        for fd in getattr(self, "_fds", {}).values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = {}

    def read(self, e: dict[str, Any], which: str = "img") -> bytes | None:
        off, size = e[f"{which}_offset"], e[f"{which}_size"]
        if off is None or size is None:
            return None
        fd = self._fds.get(e["shard"])
        if fd is None:
            fd = os.open(pack.inside(self.shard_dir, e["shard"]), os.O_RDONLY)
            self._fds[e["shard"]] = fd
        data = os.pread(fd, int(size), int(off))
        if len(data) != int(size):
            raise OSError(f"{e['shard']}: short read at {off} ({len(data)} of {size} bytes)")
        return data

    def image(self, e: dict[str, Any]) -> Image.Image:
        img = Image.open(io.BytesIO(self.read(e)))
        return img.convert("RGB")

    def mask(self, e: dict[str, Any]) -> np.ndarray:
        """Boolean edit mask (all False for an image without one)."""
        data = self.read(e, "mask")
        if data is None:
            return np.zeros((int(e["height"]), int(e["width"])), dtype=bool)
        with Image.open(io.BytesIO(data)) as m:
            return np.asarray(m.convert("L")) > 0


def eligible(
    rows: Sequence[dict[str, Any]], split: str, profile: str = "unified"
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows of ``split`` the profile trains on; edited images without a valid crop are dropped
    (returned as the second list). The detection-protocol model never sees edited images."""
    keep, dropped = [], []
    for e in rows:
        if e["split"] != split:
            continue
        if profile == "detection" and (e["source"] != "genimage" or e["label"] == "edited"):
            continue
        if e["label"] == "edited" and e.get("valid_crop") is False:
            dropped.append(e["uid"])
            continue
        keep.append(e)
    return keep, sorted(dropped)


class ForensicDataset(Dataset):
    def __init__(
        self,
        shard_dir: Path,
        split: str = "train",
        *,
        profile: str = "unified",
        crop: int = crops.CROP,
        augment: Augment | None = None,
        seed: int = 42,
        n_per_class: int | None = None,
        rows: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val, got {split!r}")
        self.reader = ShardReader(shard_dir, rows)
        self.split, self.profile, self.crop, self.seed = split, profile, crop, seed
        self.augment = Augment(enabled=False) if split == "val" else (augment or Augment())
        self.rows, self.dropped = eligible(self.reader.rows, split, profile)
        if self.dropped:
            log.info("%s: %d edited images without a valid crop left out", split, len(self.dropped))
        self.n_per_class = n_per_class
        self.epoch = -1
        self.plan: epoch.EpochPlan | None = None
        self.items: list[tuple[int, Spec]] = []
        if split == "train":
            if not n_per_class:
                raise ValueError("training needs n_per_class (crops per class per epoch)")
            self.set_epoch(0)
        else:
            self.items = self._validation_items()

    # ---------------------------------------------------------------- plans

    def epoch_seed(self, ep: int) -> int:
        return int(np.random.SeedSequence([self.seed, ep]).generate_state(1)[0])

    def set_epoch(self, ep: int) -> None:
        """Build the deterministic plan of epoch ``ep`` (training only)."""
        if self.split != "train" or ep == self.epoch:
            return
        self.plan = epoch.epoch_plan(
            [e["source"] for e in self.rows],
            [e["label"] for e in self.rows],
            [e["uid"] for e in self.rows],
            int(self.n_per_class or 0),
            self.epoch_seed(ep),
            self._train_crop,
            profile=self.profile,
        )
        self.items = self.plan.items
        self.epoch = ep

    def _train_crop(self, i: int, rng: np.random.Generator) -> Spec | None:
        e = self.rows[i]
        w, h = int(e["width"]), int(e["height"])
        s = self.augment.scale(rng, w, h, self.crop)
        sw, sh = scaled_size(w, h, s)
        if e["label"] == "edited":
            box = crops.edited_crop(resize_mask(self.reader.mask(e), s), rng, self.crop)[0]
        else:
            box = crops.random_box(sw, sh, rng, self.crop)
        return None if box is None else (*box, s)

    def _validation_items(self) -> list[tuple[int, Spec]]:
        items = []
        for i, e in enumerate(self.rows):
            rng = np.random.default_rng(_uid_seed(self.seed, e["uid"]))
            if e["label"] == "edited":
                mask = self.reader.mask(e)
                box = crops.edited_crop(mask, rng, self.crop)[0] or _valid_window(
                    mask, rng, self.crop
                )
            else:
                box = crops.random_box(int(e["width"]), int(e["height"]), rng, self.crop)
            if box is None:  # cannot happen for packed rows (sides >= crop, valid_crop True)
                self.dropped.append(e["uid"])
                continue
            items.append((i, (*box, 1.0)))
        return items

    # ---------------------------------------------------------------- items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> dict[str, Any]:
        i, (left, top, right, bottom, s) = self.items[k]
        e = self.rows[i]
        img = resize_image(self.reader.image(e), s)
        crop = np.asarray(img.crop((left, top, right, bottom)), dtype=np.uint8)
        has_mask = e["label"] == "edited"
        if has_mask:
            mask = resize_mask(self.reader.mask(e), s)[top:bottom, left:right]
        else:
            mask = np.zeros(crop.shape[:2], dtype=bool)
        rng = np.random.default_rng([self.seed, max(self.epoch, 0), k, 3])
        crop, flipped = self.augment.photometric(crop, rng)
        if flipped:
            mask = mask[:, ::-1]
        return {
            "image": torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255,
            "mask": torch.from_numpy(np.array(mask)).float().unsqueeze(0),
            "label": LABELS[e["label"]],
            "has_mask": has_mask,
            "source": e["source"],
            "uid": e["uid"],
        }


def _valid_window(
    mask: np.ndarray, rng: np.random.Generator, size: int
) -> tuple[int, int, int, int] | None:
    """A window chosen uniformly among all that pass the edited-share rule (validation only)."""
    counts = crops.window_counts(mask, size)
    lo, hi = crops.EDITED_SHARE[0] * size * size, crops.EDITED_SHARE[1] * size * size
    ok = np.argwhere((counts >= lo) & (counts <= hi))
    if ok.size == 0:
        return None
    top, left = (int(v) for v in ok[int(rng.integers(0, len(ok)))])
    return left, top, left + size, top + size
