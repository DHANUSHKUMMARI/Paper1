"""Runtime environment: Colab/Drive detection, project paths, secrets and session info.

This is the only module that may contain machine-specific paths. Everything else asks
:func:`get_paths`. Secrets are read here and nowhere else, and their values are never
logged, returned in reports or written to disk.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / "Paper 1"
DEFAULT_LOCAL_DATA = Path("/content/data")

ENV_DRIVE_ROOT = "EVIGROUND_DRIVE_ROOT"
ENV_LOCAL_DATA = "EVIGROUND_LOCAL_DATA"
ENV_DRY_RUN = "EVIGROUND_DRY_RUN"

REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingSecretError(RuntimeError):
    """Raised when a required secret is not available."""


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def dry_run() -> bool:
    """True when ``EVIGROUND_DRY_RUN=1``: no Drive mount, no downloads, temp folders only."""
    return os.environ.get(ENV_DRY_RUN) == "1"


def drive_mounted() -> bool:
    """True when Google Drive is mounted at the standard Colab location."""
    return (DRIVE_MOUNT / "MyDrive").is_dir()


@dataclass(frozen=True)
class Paths:
    """Project folders. ``root`` is persistent (Drive); ``local_data`` is fast and ephemeral."""

    root: Path
    local_data: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def shards(self) -> Path:
        return self.cache / "shards"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def ensure(self) -> Paths:
        """Create the folders this project writes to (never the datasets folder)."""
        for folder in (self.cache, self.runs, self.outputs, self.local_data):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    """Resolve project paths.

    Order: environment variables, then the Colab defaults, then a temp folder
    (local development, CI and dry runs never touch a real Drive).
    """
    fallback = Path(tempfile.gettempdir()) / "eviground"
    use_colab_defaults = in_colab() and not dry_run()

    root_env = os.environ.get(ENV_DRIVE_ROOT)
    if root_env:
        root = Path(root_env)
    elif use_colab_defaults:
        root = DEFAULT_DRIVE_ROOT
    else:
        root = fallback / "drive" / "Paper 1"

    local_env = os.environ.get(ENV_LOCAL_DATA)
    if local_env:
        local = Path(local_env)
    elif use_colab_defaults:
        local = DEFAULT_LOCAL_DATA
    else:
        local = fallback / "data"
    return Paths(root=root, local_data=local)


def get_secret(name: str) -> str:
    """Return a secret from Colab Secrets, falling back to an environment variable.

    The value is never logged. Raises :class:`MissingSecretError` naming the secret.
    """
    value: str | None = None
    if in_colab():
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            value = userdata.get(name)
        except Exception:  # secret absent or notebook access not granted
            value = None
    if not value:
        value = os.environ.get(name)
    if not value:
        raise MissingSecretError(
            f"Secret '{name}' is not set. Add it in the Colab Secrets panel (key icon) "
            "with notebook access enabled, or export it as an environment variable."
        )
    return value


@contextlib.contextmanager
def runtime_secrets(names: Sequence[str]) -> Iterator[None]:
    """Expose secrets as environment variables for the duration of the block only.

    Previous values (if any) are restored on exit, so nothing leaks into later code.
    """
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = get_secret(name)
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def drive_quota() -> dict[str, int | None] | None:
    """Google Drive storage quota in bytes, read through the Drive API (Colab only).

    On Colab, ``shutil.disk_usage`` of the Drive mount reports the virtual machine's own
    disk, not the Drive quota (seen on 12 Nov 2026: 26.5 GB written to Drive moved the
    "Drive free" figure by only 5 GB, together with the local disk). This asks Google
    instead. The first call shows Colab's sign-in prompt. Returns None outside Colab or if
    the quota cannot be read; ``free`` is None for an unlimited plan.
    """
    if not in_colab() or dry_run():
        return None
    try:
        from google.colab import auth  # type: ignore[import-not-found]
        from googleapiclient.discovery import build  # type: ignore[import-not-found]

        auth.authenticate_user()
        quota = build("drive", "v3", cache_discovery=False).about().get(
            fields="storageQuota").execute()["storageQuota"]  # fmt: skip
    except Exception as exc:  # no secret involved; report the type only
        log.warning("could not read the Google Drive quota: %s", type(exc).__name__)
        return None
    limit = int(quota["limit"]) if quota.get("limit") else None
    usage = int(quota.get("usage", 0))
    return {"limit": limit, "usage": usage, "free": None if limit is None else limit - usage}


def drive_free_bytes() -> int | None:
    """Free bytes in the Google Drive quota (None if unknown; very large if unlimited)."""
    quota = drive_quota()
    if quota is None:
        return None
    return quota["free"] if quota["free"] is not None else 1 << 62


def git_commit(repo: Path | None = None) -> str | None:
    """Short commit hash of the checked-out code, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _free_gb(path: Path) -> float | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 1e9, 1)
    except OSError:
        return None


def _gpu() -> dict[str, object]:
    info: dict[str, object] = {"torch": None, "cuda": False, "gpu_name": None, "gpu_mem_gb": None}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(cuda=True, gpu_name=props.name, gpu_mem_gb=round(props.total_memory / 1e9, 1))
    return info


def session_info(cu_per_hour: float | None = None) -> dict[str, object]:
    """Collect and log what this session runs on: GPU, Drive, free space, commit, CU rate.

    Contains no secret values, so the result is safe to print and to save in a run folder.
    """
    paths = get_paths()
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "in_colab": in_colab(),
        "dry_run": dry_run(),
        "drive_mounted": drive_mounted(),
        "drive_root": str(paths.root),
        "drive_free_gb": _free_gb(paths.root),
        "local_data": str(paths.local_data),
        "local_free_gb": _free_gb(paths.local_data),
        "git_commit": git_commit(),
        "cu_per_hour": cu_per_hour,
    }
    info.update(_gpu())
    log.info(
        "session: python %s | torch %s | gpu %s (%s GB) | drive mounted=%s free=%s GB | "
        "commit %s | CU/h %s",
        info["python"],
        info["torch"],
        info["gpu_name"] or "none",
        info["gpu_mem_gb"],
        info["drive_mounted"],
        info["drive_free_gb"],
        info["git_commit"],
        cu_per_hour,
    )
    return info
