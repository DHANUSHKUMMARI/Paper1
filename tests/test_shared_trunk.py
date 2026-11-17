"""CP4 shared trunk: the probe's features from the joint pass equal a plain frozen CLIP's."""

from __future__ import annotations

import pytest

pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from eviground.config import load_config  # noqa: E402
from eviground.models.eviground import build_model  # noqa: E402
from eviground.models.semantic import SemanticBranch, _vision  # noqa: E402

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
}


def _branch(lora_layers):
    torch.manual_seed(0)
    s = SemanticBranch(tiny=TINY, d=32, lora_rank=4, lora_layers=lora_layers).eval()
    with torch.no_grad():  # LoRA B starts at zero: make the adapters change the output
        for n, p in s.clip.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    return s


def _plain_clip(s):
    """A plain ``CLIPVisionModel`` with the branch's frozen weights (no LoRA)."""
    from transformers import CLIPVisionModel

    ref = CLIPVisionModel(s.clip.config).eval()
    state = {
        k.replace(".base_layer.", "."): v
        for k, v in s.clip.state_dict().items()
        if "lora_" not in k
    }
    ref.load_state_dict(state, strict=True)
    return ref


@pytest.mark.parametrize("lora_layers", [[2, 3], [1, 2, 3]])  # the boundary moves with the fix
def test_probe_features_equal_a_plain_frozen_clip(lora_layers):
    s = _branch(lora_layers)
    ref = _plain_clip(s)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = ref(pixel_values=x)
        want_patches = _vision(ref).post_layernorm(out.last_hidden_state)[:, 1:]
        model_tokens, cls, patches = s.joint_tokens(x)
    assert s.first_trainable == lora_layers[0]
    assert torch.equal(cls, out.pooler_output)
    assert torch.equal(patches, want_patches)
    # the model's own tokens are unchanged by the joint pass, and the adapters were active there
    assert torch.equal(model_tokens, s.clip_tokens(x))
    assert not torch.allclose(model_tokens, out.last_hidden_state, atol=1e-3)


def test_adapters_are_on_again_after_the_probe_pass():
    s = _branch([2, 3])
    x = torch.randn(1, 3, 224, 224)
    before = s.clip_tokens(x).detach()
    s.probe_features(x)
    assert torch.equal(s.clip_tokens(x).detach(), before)
    assert all(p.requires_grad for p in s.lora_parameters())


def test_model_forward_with_probe_keeps_the_model_outputs():
    torch.manual_seed(0)
    m = build_model(load_config("debug").model).eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        plain = m(x)
        joint = m(x, probe=True)
    for k in ("logits", "mask_logits", "alpha", "gate", "z"):
        assert torch.equal(plain[k], joint[k]), k
    assert joint["probe_cls"].shape == (2, 64)
    assert joint["probe_patches"].shape == (2, 256, 64)


def test_probe_features_do_not_keep_the_whole_token_tensor():
    """Notebook 08, 17 Nov 2026: the class token was a view of the (B, 257, D) token tensor, so
    every batch kept in ``train_probes`` held all its tokens: about 67 MB per batch of 64 on the
    real CLIP, an out-of-memory error on the L4 after about 320 batches."""
    from eviground.baselines import clip_probe

    torch.manual_seed(0)
    m = build_model(load_config("debug").model).eval()
    x = torch.rand(4, 3, 128, 128)
    cls, patches = clip_probe._features(m, x)
    assert cls.untyped_storage().nbytes() == cls.numel() * cls.element_size()
    xs, _ys = clip_probe.collect_cls(
        m, [{"image": x, "label": torch.zeros(4)}], torch.device("cpu")
    )
    assert xs.untyped_storage().nbytes() == xs.numel() * xs.element_size()
    assert cls.shape == (4, m.semantic.width) and patches.shape[0] == 4
