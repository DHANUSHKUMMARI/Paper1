"""Repository hygiene audit: no secrets, datasets or checkpoints in Git (checkpoint CP0).

Run ``python -m eviground.utils.repo_audit --history`` or call :func:`run_audit`.
The same checks run in the test suite and in CI.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# Model weights, archives and array dumps never belong in the repository.
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".h5", ".npz", ".npy",
    ".tar", ".gz", ".tgz", ".zip", ".7z",
}  # fmt: skip
FORBIDDEN_NAMES = {".env", "kaggle.json", ".netrc", "credentials.json", "token.json"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
IMAGE_ALLOWED_PREFIXES = ("docs/",)  # paper figures only; never dataset images
PARQUET_ALLOWED_PREFIX = "data/manifests/"
MAX_FILE_BYTES = 5 * 1024 * 1024
OFFICE_SUFFIXES = {".docx", ".xlsx", ".pptx"}

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "kaggle_api_token": re.compile(r"KGAT_[A-Za-z0-9]{20,}"),
    "kaggle_legacy_key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "github_fine_grained_token": re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    "token_in_git_url": re.compile(r"https://[A-Za-z0-9_\-]{20,}(?::[^@\s/]*)?@github\.com"),
    "api_key_sk": re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
}

MUST_BE_IGNORED = (
    "data/genimage/img_0001.jpg",
    "runs/p1_seed42/checkpoints/last.pt",
    "outputs/tables/p1.csv",
    "cache/shards/genimage/train-00000.tar",
    "model.ckpt",
    "weights.safetensors",
    "best.pth",
    ".env",
    "kaggle.json",
)
MUST_NOT_BE_IGNORED = ("data/README.md", "data/manifests/genimage_sd14.csv", "src/eviground/env.py")

REQUIRED_PATHS = (
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    ".github/workflows/ci.yml",
    "configs/base.yaml",
    "configs/debug.yaml",
    "data/README.md",
    "docs/CHECKPOINTS.md",
    "src/eviground/__init__.py",
    "src/eviground/env.py",
    "src/eviground/config.py",
    "src/eviground/utils/seed.py",
    "src/eviground/data/__init__.py",
    "src/eviground/models/__init__.py",
    "src/eviground/train/__init__.py",
    "src/eviground/eval/__init__.py",
    "src/eviground/explain/__init__.py",
    "src/eviground/baselines/__init__.py",
    "tests/conftest.py",
)


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def is_git_repo(repo: Path) -> bool:
    try:
        return _git(repo, "rev-parse", "--git-dir").returncode == 0
    except OSError:
        return False


def tracked_files(repo: Path) -> list[str]:
    out = _git(repo, "ls-files", "-z").stdout
    return [p for p in out.split("\0") if p]


def check_tracked_files(repo: Path) -> list[str]:
    """Forbidden file types, forbidden names, oversized files, stray images."""
    findings: list[str] = []
    for rel in tracked_files(repo):
        path = Path(rel)
        suffix = path.suffix.lower()
        if path.name in FORBIDDEN_NAMES:
            findings.append(f"forbidden file tracked: {rel}")
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"weights/archive file tracked: {rel}")
        if suffix == ".parquet" and not rel.startswith(PARQUET_ALLOWED_PREFIX):
            findings.append(f"parquet outside data/manifests tracked: {rel}")
        if suffix in IMAGE_SUFFIXES and not rel.startswith(IMAGE_ALLOWED_PREFIXES):
            findings.append(f"image file tracked outside docs/: {rel}")
        full = repo / rel
        if full.is_file() and full.stat().st_size > MAX_FILE_BYTES:
            findings.append(f"file larger than 5 MB tracked: {rel} ({full.stat().st_size} bytes)")
    return findings


def _texts(name: str, blob: bytes) -> list[str]:
    """Decode a blob for scanning; Office files are unzipped so their XML is scanned too."""
    texts = [blob.decode("utf-8", errors="ignore")]
    if Path(name).suffix.lower() in OFFICE_SUFFIXES:
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for member in archive.namelist():
                    texts.append(archive.read(member).decode("utf-8", errors="ignore"))
        except zipfile.BadZipFile:
            pass
    return texts


def scan_blob(name: str, blob: bytes) -> list[str]:
    """Names of the secret patterns found in one file (never the matched text)."""
    hits: list[str] = []
    for text in _texts(name, blob):
        for label, pattern in SECRET_PATTERNS.items():
            if label not in hits and pattern.search(text):
                hits.append(label)
    return hits


def scan_worktree(repo: Path) -> list[str]:
    """Secret patterns in the tracked files as checked out."""
    findings: list[str] = []
    for rel in tracked_files(repo):
        full = repo / rel
        if not full.is_file():
            continue
        for label in scan_blob(rel, full.read_bytes()):
            findings.append(f"possible secret ({label}) in {rel}")
    return findings


def scan_history(repo: Path) -> tuple[list[str], int, int]:
    """Secret patterns in every blob reachable from any ref.

    Returns (findings, commits scanned, blobs scanned).
    """
    commits = [c for c in _git(repo, "rev-list", "--all").stdout.split() if c]
    listing = _git(repo, "rev-list", "--all", "--objects").stdout.splitlines()
    findings: list[str] = []
    seen: set[str] = set()
    for line in listing:
        sha, _, name = line.partition(" ")
        if not name or sha in seen:
            continue
        if _git(repo, "cat-file", "-t", sha).stdout.strip() != "blob":
            continue
        seen.add(sha)
        blob = _git(repo, "cat-file", "blob", sha, text=False).stdout
        for label in scan_blob(name, blob):
            findings.append(f"possible secret ({label}) in history: {name} @ {sha[:8]}")
    return findings, len(commits), len(seen)


def check_gitignore(repo: Path) -> list[str]:
    """Functional check: sample data, checkpoint and secret paths must be ignored."""
    findings: list[str] = []
    for rel in MUST_BE_IGNORED:
        if _git(repo, "check-ignore", "-q", "--no-index", rel).returncode != 0:
            findings.append(f".gitignore does not ignore: {rel}")
    for rel in MUST_NOT_BE_IGNORED:
        if _git(repo, "check-ignore", "-q", "--no-index", rel).returncode == 0:
            findings.append(f".gitignore wrongly ignores: {rel}")
    return findings


def check_structure(repo: Path) -> list[str]:
    return [f"missing required path: {rel}" for rel in REQUIRED_PATHS if not (repo / rel).exists()]


def run_audit(repo: Path | str, history: bool = False) -> dict[str, object]:
    """Run every check and return a JSON-serialisable report (``ok`` is the verdict)."""
    repo = Path(repo).resolve()
    if not is_git_repo(repo):
        raise RuntimeError(f"not a git repository: {repo}")
    report: dict[str, object] = {
        "repo": str(repo),
        "tracked_files": len(tracked_files(repo)),
        "tracked_file_findings": check_tracked_files(repo),
        "secret_findings": scan_worktree(repo),
        "gitignore_findings": check_gitignore(repo),
        "structure_findings": check_structure(repo),
        "history_scanned": history,
    }
    if history:
        findings, commits, blobs = scan_history(repo)
        report.update(history_findings=findings, history_commits=commits, history_blobs=blobs)
    keys = [k for k in report if k.endswith("_findings")]
    report["ok"] = all(not report[k] for k in keys)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="repository root (default: current directory)")
    parser.add_argument("--history", action="store_true", help="also scan every blob in history")
    args = parser.parse_args(argv)
    report = run_audit(args.repo, history=args.history)
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
