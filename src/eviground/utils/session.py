"""Lean GPU sessions (Dhanush, 18 Nov 2026: compute is the constraint).

Every GPU notebook:

1. runs :func:`preflight` before any staging: it prints what will run (model, experiment, result
   file), the estimated compute units and the balance after it, checks that the inputs exist, and
   refuses when the estimate exceeds the balance;
2. ends with :func:`finish` (writes ``compute.json`` for the session) and :func:`release`
   (``google.colab.runtime.unassign()``), after its outputs are saved.

Outside Colab and in the dry run, :func:`release` does nothing and says so.
"""

from __future__ import annotations

import datetime
import importlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PreflightError(RuntimeError):
    """A GPU run whose inputs or budget do not match what was asked for."""


def preflight(
    what: Mapping[str, Any],
    *,
    paths: Mapping[str, Path] | None = None,
    est_cu: float,
    cu_left: float | None = None,
    expect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check and print a run before it starts.

    ``what``: the run's settings to print (model, experiment, result file ...).
    ``paths``: inputs that must exist (checkpoints, frozen files).
    ``expect``: settings that must equal the given value (``what[key] == value``).
    ``est_cu``: the estimate; ``cu_left``: the balance read in Colab (None: not checked).
    """
    problems = [f"missing {k}: {p}" for k, p in (paths or {}).items() if not Path(p).exists()]
    problems += [
        f"{k} is {what.get(k)!r}, expected {v!r}"
        for k, v in (expect or {}).items()
        if what.get(k) != v
    ]
    if cu_left is not None and est_cu > cu_left:
        problems.append(f"estimated {est_cu} CU exceeds the {cu_left} CU left")
    after = None if cu_left is None else round(cu_left - est_cu, 2)
    report = {
        "run": dict(what),
        "est_cu": est_cu,
        "cu_left_before": cu_left,
        "cu_left_after": after,
    }
    for k, v in what.items():
        print(f"  {k}: {v}")
    print(f"  estimated CU: {est_cu}" + ("" if after is None else f" | CU left after: {after}"))
    if problems:
        raise PreflightError("; ".join(problems))
    return report


def finish(
    folder: Path,
    t0: float,
    cu_per_hour: float,
    *,
    name: str = "compute.json",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the session's ``compute.json`` (wall hours x rate) into ``folder``."""
    hours = (time.time() - t0) / 3600
    info = {
        "wall_time_h": round(hours, 4),
        "cu_per_hour": cu_per_hour,
        "compute_units": round(hours * cu_per_hour, 3),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **(dict(extra) if extra else {}),
    }
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(info, indent=1, default=str))
    hours_cu = f"{info['wall_time_h']} h x {cu_per_hour} = {info['compute_units']} CU"
    print(f"compute: {hours_cu} -> {folder / name}")
    return info


def release(dry: bool) -> str:
    """Disconnect the Colab runtime so it stops using compute units. Call it last."""
    if dry:
        print("release: dry run, runtime kept")
        return "dry"
    try:
        colab = importlib.import_module("google.colab")
    except ImportError:
        print("release: not in Colab, nothing to do")
        return "not_colab"
    print("release: unassigning the runtime (outputs are saved)")
    colab.runtime.unassign()
    return "unassigned"
