"""Checkpoints (Milestone 3 plan, step 3, with Dhanush's fixes of 14 Nov 2026).

What a checkpoint holds:

- the full ``state_dict`` of every module except the frozen CLIP: LoRA, projection, trace branch,
  fusion and heads, **buffers included** (EfficientNet BatchNorm running statistics and
  ``num_batches_tracked``, the fixed view kernels). The frozen CLIP is reloaded from the Hugging
  Face cache at the pinned revision; its SHA-256 is stored and checked on load;
- optimizer, scheduler and (fp16) grad-scaler states;
- the run position (epoch, step in epoch, global step), best score and patience counter;
- the Python, NumPy, torch CPU and CUDA (all devices) RNG states;
- the CLIP revision, the frozen-CLIP SHA-256, the shard index version and the mask-rule version,
  the config and the git commit.

Writes are atomic: ``<name>.tmp`` then ``os.replace``, so a run killed mid-save keeps the
previous file.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

FROZEN_PREFIX = "semantic.clip."


class CheckpointMismatch(RuntimeError):
    """The frozen CLIP or the data of a checkpoint differs from the current run: stop."""


def is_frozen_clip(name: str) -> bool:
    """State-dict keys of the frozen CLIP (everything under ``semantic.clip`` except LoRA)."""
    return name.startswith(FROZEN_PREFIX) and "lora_" not in name


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter and buffer except the frozen CLIP, on the CPU."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not is_frozen_clip(k)}


def frozen_clip_sha256(model: nn.Module) -> str:
    """SHA-256 over the frozen CLIP state: names, dtypes, shapes and bytes, in sorted order.
    Computed once per run (and once on load for the check)."""
    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(k for k in state if is_frozen_clip(k)):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _numpy_state() -> dict[str, Any]:
    """NumPy's RNG state as plain values and a tensor (loadable with ``weights_only=True``)."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "name": str(name),
        "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached": float(cached),
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _numpy_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"])
    )
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path: Path, state: dict[str, Any]) -> None:
    """Atomic save: a crash leaves either the old file or the new one, never half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def copy(src: Path, dst: Path) -> None:
    """Atomic copy (``last.pt`` from a step file)."""
    tmp = Path(dst).with_name(Path(dst).name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def load(path: Path) -> dict[str, Any]:
    """Tensors and plain Python values only (``weights_only=True``): loading a checkpoint never
    runs code from the file."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def restore_model(
    model: nn.Module, state: dict[str, Any], *, clip_revision: Any, clip_sha256: str
) -> None:
    """Load a checkpoint's model state after checking the frozen CLIP (revision and SHA-256)."""
    if state.get("clip_revision") != clip_revision:
        raise CheckpointMismatch(
            f"CLIP revision {clip_revision!r} differs from the checkpoint's "
            f"{state.get('clip_revision')!r}: stop"
        )
    if state.get("clip_sha256") != clip_sha256:
        raise CheckpointMismatch(
            "the frozen CLIP weights differ from the ones this checkpoint was trained with "
            f"(sha256 {clip_sha256[:12]} vs {str(state.get('clip_sha256'))[:12]}): stop"
        )
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    missing = [k for k in missing if not is_frozen_clip(k)]
    if missing or unexpected:
        raise CheckpointMismatch(f"model keys differ: missing {missing}, unexpected {unexpected}")


STEP_FILE = re.compile(r"step_(\d+)\.pt$")


def rotate(run_dir: Path, keep: int = 2) -> list[Path]:
    """Delete all but the newest ``keep`` step files; returns the ones removed."""
    steps = sorted(
        (p for p in Path(run_dir).glob("step_*.pt") if STEP_FILE.search(p.name)),
        key=lambda p: int(STEP_FILE.search(p.name).group(1)),  # type: ignore[union-attr]
    )
    gone = steps[: max(0, len(steps) - keep)]
    for p in gone:
        p.unlink()
    return gone
