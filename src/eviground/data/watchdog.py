"""A process pool that survives reads that never return.

On Colab, a read from the Google Drive mount can block forever (13 Nov 2026: the CP1 scan of
GenImage Validation, read in place on Drive, stopped after a few thousand images on every run).
A plain process pool then waits forever. :func:`run_watched` knows which item each worker is
working on; a worker busy with one item for more than ``stall_s`` seconds is killed and
replaced, and the item is retried. An item that stalls ``retries`` times gets ``on_timeout``
as its result, so the run always finishes and the stall is reported, not hidden.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

log = logging.getLogger(__name__)


def _worker(func: Callable[[Any], Any], tasks: Any, results: Any, wid: int) -> None:
    while True:
        task = tasks.get()
        if task is None:
            return
        idx, item = task
        results.put(("start", wid, idx, None))
        try:
            out = func(item)
        except Exception as exc:  # the caller decides; a crash must not lose the item
            out = exc
        results.put(("done", wid, idx, out))


def run_watched(
    func: Callable[[Any], Any],
    items: Sequence[Any],
    workers: int,
    *,
    stall_s: float = 180.0,
    retries: int = 2,
    on_timeout: Callable[[Any], Any],
    progress: Callable[[int, int], None] | None = None,
    describe: Callable[[Any], str] = str,
    poll_s: float = 1.0,
) -> list[Any]:
    """``[func(item) for item in items]`` on ``workers`` processes, in input order.

    A worker that raises returns the exception object as that item's result. ``progress(done,
    total)`` is called after each finished item.
    """
    n = len(items)
    if n == 0:
        return []
    ctx = mp.get_context("fork")
    tasks, results = ctx.Queue(), ctx.Queue()
    for idx in range(n):
        tasks.put((idx, items[idx]))
    procs: dict[int, Any] = {}
    current: dict[int, tuple[int, float]] = {}  # worker id -> (item, start time)
    attempts: Counter[int] = Counter()
    out: dict[int, Any] = {}
    next_wid = 0

    def spawn() -> None:
        nonlocal next_wid
        p = ctx.Process(target=_worker, args=(func, tasks, results, next_wid), daemon=True)
        p.start()
        procs[next_wid] = p
        next_wid += 1

    def give_up_or_retry(idx: int, why: str) -> None:
        attempts[idx] += 1
        if attempts[idx] >= retries:
            log.warning(
                "%s: %s %d times, recorded as a timeout", describe(items[idx]), why, retries
            )
            out[idx] = on_timeout(items[idx])
            if progress:
                progress(len(out), n)
        else:
            log.warning(
                "%s: %s, retrying (%d of %d)", describe(items[idx]), why, attempts[idx], retries
            )
            tasks.put((idx, items[idx]))

    for _ in range(min(workers, n)):
        spawn()
    last_event = time.monotonic()
    try:
        while len(out) < n:
            try:
                kind, wid, idx, value = results.get(timeout=poll_s)
                last_event = time.monotonic()
                if kind == "start":
                    current[wid] = (idx, time.monotonic())
                else:
                    current.pop(wid, None)
                    if idx not in out:
                        out[idx] = value
                        if progress:
                            progress(len(out), n)
                continue
            except queue.Empty:
                pass
            now = time.monotonic()
            for wid, (idx, t0) in list(current.items()):
                if now - t0 > stall_s:  # stuck on this item: kill the worker, replace it
                    procs[wid].kill()
                    procs[wid].join(5)
                    del procs[wid], current[wid]
                    spawn()
                    if idx not in out:
                        give_up_or_retry(idx, f"no result after {stall_s:.0f} s")
            for wid, p in list(procs.items()):  # a worker that died on its own
                if not p.is_alive():
                    del procs[wid]
                    spawn()
                    if wid in current:
                        idx, _ = current.pop(wid)
                        if idx not in out:
                            give_up_or_retry(idx, f"worker died (exit code {p.exitcode})")
            if not current and now - last_event > 2 * stall_s:
                # nothing running and nothing heard: an item was lost between queue and start
                for idx in range(n):
                    if idx not in out:
                        tasks.put((idx, items[idx]))
                last_event = now
    finally:
        for _ in procs:
            tasks.put(None)
        for p in procs.values():
            p.join(2)
            if p.is_alive():
                p.kill()
                p.join(2)
        tasks.close()
        results.close()
    return [out[i] for i in range(n)]
