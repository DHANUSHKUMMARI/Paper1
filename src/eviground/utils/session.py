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


# Colab compute units per hour by runtime type. L4 and T4 are measured from Dhanush's balance
# (notebook 13: about 2.2 CU/h on the L4; notebook 15: 1.71 CU over 1.3405 h on a T4); the CPU
# rate is Colab's listed one. The balance drop, when known, always wins over these rates.
CU_RATES = {"L4": 2.2, "T4": 1.28, "cpu": 0.08}


def gpu_type(name: str | None) -> str:
    """'L4', 'T4', 'A100' ... from a CUDA device name; 'cpu' when there is none."""
    if not name:
        return "cpu"
    for key in ("L4", "T4", "A100", "V100", "H100"):
        if key in str(name).upper().split() or f" {key}" in str(name).upper():
            return key
    return str(name)


def cu_rate(gpu: str | None) -> float | None:
    """The CU/h rate for a GPU name or type, or None when it is not known."""
    return CU_RATES.get(gpu_type(gpu))


def _cu_fields(hours: float, rate: float | None, start: float | None, end: float | None):
    estimate = None if rate is None else round(hours * rate, 3)
    measured = None if start is None or end is None else round(start - end, 3)
    if measured is not None:
        return estimate, measured, measured, "balance"
    return estimate, None, estimate, "rate x wall time"


def finish(
    folder: Path,
    t0: float,
    cu_per_hour: float | None = None,
    *,
    name: str = "compute.json",
    extra: Mapping[str, Any] | None = None,
    gpu: str | None = None,
    balance_start: float | None = None,
    balance_end: float | None = None,
) -> dict[str, Any]:
    """Write the session's ``compute.json`` into ``folder``.

    The rate comes from the GPU type (``gpu``: the CUDA device name; :data:`CU_RATES`), or from
    ``cu_per_hour`` when the type is not known. ``balance_start`` / ``balance_end``: the Colab
    balance typed in before and after the run; when both are known their difference is the
    session's CU (``cu``, ``cu_source='balance'``), since rate x hours misses interrupted
    sessions (19 Nov) and the rate differs by GPU (24 Nov: notebook 15 ran on a T4). The end
    balance can be added later with :func:`record_balance`."""
    hours = (time.time() - t0) / 3600
    rate = cu_rate(gpu) if gpu is not None else None
    if rate is None:
        rate = cu_per_hour
    if rate is None:
        print(f"compute: no CU rate known for {gpu!r}; only the balance can give the CU")
    estimate, measured, cu, source = _cu_fields(hours, rate, balance_start, balance_end)
    info = {
        "wall_time_h": round(hours, 4),
        "gpu": gpu,
        "gpu_type": gpu_type(gpu) if gpu is not None else None,
        "cu_per_hour": rate,
        "cu_estimate": estimate,
        "cu_balance_start": balance_start,
        "cu_balance_end": balance_end,
        "cu_measured": measured,
        "cu": cu,
        "cu_source": source,
        "compute_units": cu,  # the older key, for readers written before 24 Nov 2026
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **(dict(extra) if extra else {}),
    }
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(info, indent=1, default=str))
    print(
        f"compute: {info['wall_time_h']} h on {info['gpu_type'] or 'this runtime'} at {rate} CU/h "
        f"-> {cu} CU ({source}) -> {folder / name}"
    )
    return info


def record_balance(path: Path, balance_end: float) -> dict[str, Any]:
    """Add the balance read after the run to a ``compute.json``; the measured CU then replaces the
    rate estimate."""
    path = Path(path)
    info = json.loads(path.read_text())
    _, measured, cu, source = _cu_fields(
        info.get("wall_time_h", 0.0), None, info.get("cu_balance_start"), balance_end
    )
    if measured is None:
        raise ValueError(f"{path} has no cu_balance_start; the session's CU cannot be measured")
    info.update(
        cu_balance_end=balance_end, cu_measured=measured, cu=cu, cu_source=source, compute_units=cu
    )
    path.write_text(json.dumps(info, indent=1, default=str))
    print(f"compute: balance {info['cu_balance_start']} -> {balance_end}: {measured} CU measured")
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
