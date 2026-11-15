"""The pinned CLIP weights are checked before they are loaded (15 Nov 2026, after the first CP2
run). No network: a fake Hugging Face cache with a tiny CLIP, laid out like the real one
(``snapshots/<rev>/<file>`` are links into ``blobs/``)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402

from eviground.models import semantic  # noqa: E402

REPO = "openai/clip-vit-large-patch14"
REV = "0" * 40
TINY = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """A cache holding a tiny CLIP at REV; ``hf_hub_download`` resolves into it, offline."""
    torch.manual_seed(0)
    model = semantic.load_clip(tiny=TINY)
    saved = tmp_path / "saved"
    model.save_pretrained(saved, safe_serialization=True)
    root = tmp_path / "cache" / "models--openai--clip-vit-large-patch14"
    snap = root / "snapshots" / REV
    blobs = root / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for name in ("config.json", "model.safetensors"):
        blob = blobs / _sha(saved / name)
        blob.write_bytes((saved / name).read_bytes())
        os.symlink(os.path.relpath(blob, snap), snap / name)
    calls = []

    def hf_hub_download(repo_id, filename, *, revision=None, cache_dir=None, **_kw):
        calls.append((repo_id, filename, revision))
        assert repo_id == REPO
        assert revision == REV
        return str(snap / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    semantic._SHA_CACHE.clear()
    return {"model": model, "snap": snap, "sha": _sha(saved / "model.safetensors"), "calls": calls}


def test_matching_weights_load_from_the_checked_snapshot(fake_hub, monkeypatch):
    from transformers import CLIPVisionModel

    loaded_from = []
    original = CLIPVisionModel.from_pretrained.__func__

    def spy(cls, path, *a, **k):
        loaded_from.append(Path(path))
        return original(cls, path, *a, **k)

    monkeypatch.setattr(CLIPVisionModel, "from_pretrained", classmethod(spy))
    clip = semantic.load_clip(REPO, revision=REV, weights_sha256=fake_hub["sha"])
    assert loaded_from == [fake_hub["snap"]]  # the folder whose file was checked
    want = fake_hub["model"].state_dict()
    got = clip.state_dict()
    assert got.keys() == want.keys()
    for name, a in got.items():
        assert torch.equal(a, want[name]), name
    assert (REPO, "model.safetensors", REV) in fake_hub["calls"]


def test_wrong_weights_stop_before_loading(fake_hub, monkeypatch):
    from transformers import CLIPVisionModel

    def must_not_load(*_a, **_k):
        raise AssertionError("loaded although the hash differs")

    monkeypatch.setattr(CLIPVisionModel, "from_pretrained", must_not_load)
    with pytest.raises(semantic.WeightsMismatch, match="not the pinned"):
        semantic.load_clip(REPO, revision=REV, weights_sha256="f" * 64)


def test_a_drive_link_text_file_is_caught(fake_hub):
    """What Google Drive stored on 15 Nov: the 79-byte link text instead of the weights."""
    weights = fake_hub["snap"] / "model.safetensors"
    target = os.readlink(weights)
    weights.unlink()
    weights.write_text(target.ljust(79, "0")[:79])
    assert weights.stat().st_size == 79
    with pytest.raises(semantic.WeightsMismatch, match=r"\(79 bytes\)"):
        semantic.load_clip(REPO, revision=REV, weights_sha256=fake_hub["sha"])


def test_the_hash_is_computed_once_per_file_version(fake_hub, monkeypatch):
    semantic.load_clip(REPO, revision=REV, weights_sha256=fake_hub["sha"])
    assert len(semantic._SHA_CACHE) == 1

    def boom(*_a, **_k):
        raise AssertionError("hashed again")

    monkeypatch.setattr(semantic.hashlib, "sha256", boom)
    semantic.load_clip(REPO, revision=REV, weights_sha256=fake_hub["sha"])


def test_config_pins_the_commit_and_the_weights():
    from eviground.config import load_config

    sem = load_config("base").model.semantic
    assert sem.revision == "32bd64288804d66eefd0ccbe215aa642df71cc41"
    assert sem.weights_sha256 == (
        "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"
    )


def test_trainer_refuses_a_real_backbone_without_the_weights_hash(tmp_path):
    from eviground.config import load_config
    from eviground.train.trainer import Trainer

    cfg = load_config("debug")
    cfg.model.semantic.tiny = None
    cfg.model.semantic.revision = REV
    cfg.model.semantic.weights_sha256 = None
    with pytest.raises(ValueError, match="weights_sha256"):
        Trainer(cfg, tmp_path / "shards", tmp_path / "run", log=None)
