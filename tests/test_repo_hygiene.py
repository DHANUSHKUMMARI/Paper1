"""CP0: the repository holds no secrets, datasets or checkpoints, and its layout is intact."""

from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path

import pytest

from eviground.utils import repo_audit

needs_git = pytest.mark.skipif(
    not repo_audit.is_git_repo(Path(__file__).resolve().parents[1]),
    reason="not running from a git checkout",
)


@needs_git
def test_no_forbidden_or_oversized_files_tracked(repo_root: Path) -> None:
    assert repo_audit.check_tracked_files(repo_root) == []


@needs_git
def test_no_secrets_in_tracked_files(repo_root: Path) -> None:
    assert repo_audit.scan_worktree(repo_root) == []


@needs_git
def test_no_secrets_in_git_history(repo_root: Path) -> None:
    findings, commits, blobs = repo_audit.scan_history(repo_root)
    assert commits >= 1 and blobs >= 1
    assert findings == []


@needs_git
def test_gitignore_blocks_data_checkpoints_and_secrets(repo_root: Path) -> None:
    assert repo_audit.check_gitignore(repo_root) == []


def test_required_structure_exists(repo_root: Path) -> None:
    assert repo_audit.check_structure(repo_root) == []


# --- the scanner itself must catch what it claims to catch -------------------------------


def _fake(prefix: str, length: int) -> str:
    """Build a token-shaped string at runtime so no such literal exists in this file."""
    return prefix + "A1b2C3d4" * (length // 8 + 1)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("huggingface_token", _fake("hf_", 34)),
        ("kaggle_api_token", _fake("KGAT_", 32)),
        ("kaggle_legacy_key", '{"username": "someone", "key": "' + "0a1b2c3d" * 4 + '"}'),
        ("github_token", _fake("ghp_", 36)),
        ("github_fine_grained_token", _fake("github_pat_", 60)),
        ("token_in_git_url", "https://" + _fake("ghp_", 36) + "@github.com/user/repo.git"),
        ("private_key_block", "-----BEGIN " + "OPENSSH PRIVATE KEY-----"),
    ],
)
def test_scanner_detects_planted_secrets(label: str, text: str) -> None:
    assert label in repo_audit.scan_blob("notes.txt", f"x = '{text}'\n".encode())


def test_scanner_reads_inside_office_files() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", f"<w:t>{_fake('hf_', 34)}</w:t>")
    assert "huggingface_token" in repo_audit.scan_blob("brief.docx", buffer.getvalue())


def test_scanner_ignores_safe_text() -> None:
    safe = "tok = userdata.get('GITHUB_TOKEN')\nurl = f'https://{tok}@github.com/u/r.git'\n"
    assert repo_audit.scan_blob("nb.py", safe.encode()) == []


def test_audit_flags_a_dirty_repository(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "last.pt").write_bytes(b"weights")
    (tmp_path / "kaggle.json").write_text('{"username": "u", "key": "' + "0a1b2c3d" * 4 + '"}')
    (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff")
    git("add", "-A")
    git("commit", "-q", "-m", "dirty")
    report = repo_audit.run_audit(tmp_path, history=True)
    assert report["ok"] is False
    tracked = "\n".join(report["tracked_file_findings"])
    assert "last.pt" in tracked and "kaggle.json" in tracked and "img.jpg" in tracked
    assert any("kaggle_legacy_key" in f for f in report["secret_findings"])
    assert any("kaggle_legacy_key" in f for f in report["history_findings"])
    assert report["gitignore_findings"]  # no .gitignore at all
