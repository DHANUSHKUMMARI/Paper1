"""Environment helpers: paths, secrets, session info."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eviground import env


def test_paths_follow_environment(clean_env: Path) -> None:
    paths = env.get_paths()
    assert paths.root == clean_env
    assert paths.datasets == clean_env / "datasets"
    assert paths.shards == clean_env / "cache" / "shards"
    assert paths.runs == clean_env / "runs"
    assert paths.outputs == clean_env / "outputs"


def test_ensure_creates_writable_folders_but_not_datasets(clean_env: Path) -> None:
    paths = env.get_paths().ensure()
    assert paths.runs.is_dir() and paths.outputs.is_dir() and paths.cache.is_dir()
    assert paths.local_data.is_dir()
    assert not paths.datasets.exists()


def test_paths_never_default_to_real_drive_outside_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.ENV_DRIVE_ROOT, raising=False)
    monkeypatch.delenv(env.ENV_LOCAL_DATA, raising=False)
    monkeypatch.setattr(env, "in_colab", lambda: False)
    paths = env.get_paths()
    assert not str(paths.root).startswith(str(env.DRIVE_MOUNT))
    assert not str(paths.local_data).startswith("/content")


def test_dry_run_overrides_colab_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.ENV_DRIVE_ROOT, raising=False)
    monkeypatch.setattr(env, "in_colab", lambda: True)
    monkeypatch.setenv(env.ENV_DRY_RUN, "1")
    assert env.dry_run()
    assert not str(env.get_paths().root).startswith(str(env.DRIVE_MOUNT))


def test_colab_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (env.ENV_DRIVE_ROOT, env.ENV_LOCAL_DATA, env.ENV_DRY_RUN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env, "in_colab", lambda: True)
    paths = env.get_paths()
    assert paths.root == Path("/content/drive/MyDrive/Paper 1")
    assert paths.local_data == Path("/content/data")


def test_missing_secret_names_the_secret(clean_env: Path) -> None:
    with pytest.raises(env.MissingSecretError, match="HF_TOKEN"):
        env.get_secret("HF_TOKEN")


def test_secret_from_environment(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy-value")
    assert env.get_secret("HF_TOKEN") == "dummy-value"


def test_runtime_secrets_are_removed_afterwards(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "get_secret", lambda name: f"value-of-{name}")
    with env.runtime_secrets(["HF_TOKEN", "KAGGLE_KEY"]):
        assert os.environ["HF_TOKEN"] == "value-of-HF_TOKEN"
        assert os.environ["KAGGLE_KEY"] == "value-of-KAGGLE_KEY"
    assert "HF_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ


def test_runtime_secrets_cleanup_on_error_and_restore(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "pre-existing")
    monkeypatch.setattr(env, "get_secret", lambda name: "temporary")
    with pytest.raises(RuntimeError, match="boom"):
        with env.runtime_secrets(["HF_TOKEN", "KAGGLE_KEY"]):
            raise RuntimeError("boom")
    assert "HF_TOKEN" not in os.environ
    assert os.environ["KAGGLE_KEY"] == "pre-existing"


def test_runtime_secrets_missing_secret_sets_nothing(clean_env: Path) -> None:
    with pytest.raises(env.MissingSecretError):
        with env.runtime_secrets(["HF_TOKEN"]):
            pass
    assert "HF_TOKEN" not in os.environ


def test_session_info_is_complete_and_leaks_no_secrets(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "super-secret-value-123")
    monkeypatch.setenv("GITHUB_TOKEN", "another-secret-456")
    info = env.session_info(cu_per_hour=1.7)
    expected = {
        "python", "platform", "in_colab", "dry_run", "drive_mounted", "drive_root",
        "drive_free_gb", "local_data", "local_free_gb", "git_commit", "cu_per_hour",
        "torch", "cuda", "gpu_name", "gpu_mem_gb",
    }  # fmt: skip
    assert expected <= set(info)
    assert info["cu_per_hour"] == 1.7
    dumped = json.dumps(info)
    assert "super-secret-value-123" not in dumped and "another-secret-456" not in dumped
