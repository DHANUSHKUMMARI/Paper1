"""The watched process pool: stuck reads are retried, then recorded; the run always ends."""

from __future__ import annotations

import os
import time
from pathlib import Path

from eviground.data.watchdog import run_watched


def _work(item):
    """Items: ("ok", x) -> x*2; ("hang_once", marker, x); ("hang",); ("die_once", marker, x);
    ("raise",)."""
    kind = item[0]
    if kind == "ok":
        return item[1] * 2
    if kind in ("hang_once", "die_once"):
        marker = Path(item[1])
        if not marker.exists():
            marker.write_text("x")
            if kind == "hang_once":
                time.sleep(3600)
            os._exit(3)  # the worker process dies without a result
        return item[2] * 2
    if kind == "hang":
        time.sleep(3600)
    raise ValueError("bad item")


def _timeout(item):
    return "TIMEOUT"


def test_all_ok_in_order():
    items = [("ok", i) for i in range(40)]
    seen = []
    got = run_watched(
        _work, items, 4, stall_s=5, on_timeout=_timeout, progress=lambda d, n: seen.append((d, n))
    )
    assert got == [i * 2 for i in range(40)]
    assert seen[-1] == (40, 40) and len(seen) == 40


def test_hang_once_is_retried_and_recovered(tmp_path):
    items = [("ok", 1), ("hang_once", str(tmp_path / "m"), 7), ("ok", 3)]
    t = time.monotonic()
    got = run_watched(_work, items, 2, stall_s=1.0, retries=2, on_timeout=_timeout, poll_s=0.2)
    assert got == [2, 14, 6]
    assert time.monotonic() - t < 20


def test_always_hanging_item_becomes_timeout_and_run_finishes():
    items = [("ok", 1), ("hang",), ("ok", 5)]
    t = time.monotonic()
    got = run_watched(_work, items, 2, stall_s=1.0, retries=2, on_timeout=_timeout, poll_s=0.2)
    assert got == [2, "TIMEOUT", 10]
    assert time.monotonic() - t < 20


def test_dead_worker_is_replaced_and_item_retried(tmp_path):
    items = [("die_once", str(tmp_path / "d"), 4), ("ok", 2)]
    got = run_watched(_work, items, 1, stall_s=5, retries=2, on_timeout=_timeout, poll_s=0.2)
    assert got == [8, 4]


def test_exception_is_returned_not_raised():
    got = run_watched(_work, [("raise",), ("ok", 1)], 2, stall_s=5, on_timeout=_timeout)
    assert isinstance(got[0], ValueError) and got[1] == 2


def test_empty():
    assert run_watched(_work, [], 3, on_timeout=_timeout) == []
