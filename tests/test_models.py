"""M1-M4 model and losses (Milestone 3, PR A). CPU, tiny random CLIP, no downloads."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402

from eviground.config import load_config  # noqa: E402
from eviground.data import tiles  # noqa: E402
from eviground.models.eviground import EDITED, build_model, disable_dropout  # noqa: E402
from eviground.models.semantic import SemanticBranch  # noqa: E402
from eviground.models.views import ForensicViews, dct_matrix  # noqa: E402
from eviground.train.losses import attention_loss, dice_loss, eviground_loss  # noqa: E402

VIT_L = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(load_config("debug").model)


def _batch(size=128):
    x = torch.rand(3, 3, size, size)
    labels = torch.tensor([0, 1, 2])
    masks = torch.zeros(3, 1, size, size)
    masks[2, :, size // 4 : size // 2, size // 4 : size // 2] = 1
    return x, labels, masks


# --- M1 views -------------------------------------------------------------------------------


def test_views_shapes_and_full_resolution():
    v = ForensicViews()(torch.rand(2, 3, 256, 256))
    assert v["rgb"].shape == (2, 3, 224, 224)
    assert v["noise"].shape == (2, 3, 256, 256)
    assert v["freq"].shape == (2, 1, 256, 256)
    assert v["trace"].shape == (2, 4, 256, 256)
    assert v["noise"].abs().max() <= 1.0


def test_dct_is_orthonormal_and_a_constant_block_has_only_dc():
    d = dct_matrix(8)
    assert torch.allclose(d @ d.T, torch.eye(8), atol=1e-6)
    coef = d @ torch.full((8, 8), 7.0) @ d.T
    assert abs(float(coef[0, 0]) - 56.0) < 1e-4  # 7 * 8 for the orthonormal DC term
    coef[0, 0] = 0
    assert float(coef.abs().max()) < 1e-4


def test_srm_of_a_constant_image_is_zero_and_views_are_deterministic():
    views = ForensicViews()
    const = torch.full((1, 3, 64, 64), 0.4)
    r = views.noise_view(views.grey(const))
    assert float(r[..., 2:-2, 2:-2].abs().max()) < 1e-5  # away from the zero-padded border
    x = torch.rand(1, 3, 64, 64)
    a, b = views(x), views(x)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_freq_view_handles_sides_that_are_not_multiples_of_8():
    f = ForensicViews().freq_view(torch.rand(1, 1, 30, 21))
    assert f.shape == (1, 1, 30, 21)
    assert float(f[..., 24:, :].abs().max()) == 0.0  # beyond the last full block
    assert float(f[..., :, 16:].abs().max()) == 0.0


# --- M2 semantic ----------------------------------------------------------------------------


def test_semantic_only_lora_and_projection_train():
    s = SemanticBranch(
        tiny={
            **VIT_L,
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
        },
        d=32,
        lora_rank=4,
        lora_layers=[2, 3],
    )
    names = [n for n, p in s.named_parameters() if p.requires_grad]
    assert names
    assert all("lora_" in n or n.startswith("proj.") for n in names)
    assert s(torch.randn(2, 3, 224, 224)).shape == (2, 257, 32)


def test_semantic_trainable_parameters_under_5m_at_vit_l_width():
    with torch.device("meta"):  # real ViT-L/14 sizes, no weights allocated or downloaded
        s = SemanticBranch(tiny=VIT_L, d=512)
    assert s.trainable_count() < 5_000_000
    assert sum(p.numel() for p in s.lora_parameters()) == 4 * 2 * (8 * 1024 * 2)


def test_frozen_layers_give_the_same_tokens_as_the_full_clip():
    s = SemanticBranch(
        tiny={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
        },
        d=32,
        lora_rank=4,
        lora_layers=[2, 3],
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        full = s.clip(pixel_values=x).last_hidden_state
        assert torch.allclose(s.clip_tokens(x), full, atol=1e-5)


# --- M3, M4 and the full model --------------------------------------------------------------


@pytest.mark.parametrize("size", [128, 256, 512])
def test_model_output_shapes(model, size):
    out = model(torch.rand(2, 3, size, size))
    assert out["logits"].shape == (2, 3)
    assert out["mask_logits"].shape == (2, 1, size, size)
    assert out["alpha"].shape == (2, 256)
    assert out["gate"].shape == (2, 1)
    assert torch.allclose(out["alpha"].sum(1), torch.ones(2), atol=1e-5)
    assert bool(((out["gate"] > 0) & (out["gate"] < 1)).all())


def test_only_the_planned_parameters_require_grad(model):
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    clip_trainable = [n for n in trainable if n.startswith("semantic.clip.")]
    assert clip_trainable
    assert all("lora_" in n for n in clip_trainable)
    for prefix in ("semantic.proj.", "trace.", "fusion.", "cls_head.", "decoder."):
        assert any(n.startswith(prefix) for n in trainable), prefix


def test_frozen_clip_weights_unchanged_after_an_optimizer_step(model):
    frozen = {
        n: p.detach().clone()
        for n, p in model.semantic.clip.named_parameters()
        if not p.requires_grad
    }
    lora_before = {
        n: p.detach().clone() for n, p in model.semantic.clip.named_parameters() if p.requires_grad
    }
    opt = torch.optim.AdamW(model.param_groups(1e-3, 1e-3, 0.0))
    x, labels, masks = _batch()
    model.train()
    eviground_loss(model(x), labels, masks)["total"].backward()
    opt.step()
    for n, p in model.semantic.clip.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    moved = [
        n
        for n, p in model.semantic.clip.named_parameters()
        if n in lora_before and not torch.equal(p, lora_before[n])
    ]
    assert moved  # LoRA did learn


def test_backward_reaches_the_trace_first_conv_and_all_heads(model):
    model.zero_grad()
    x, labels, masks = _batch()
    eviground_loss(model(x), labels, masks)["total"].backward()
    assert model.trace.backbone.conv_stem.weight.grad.abs().sum() > 0
    assert model.cls_head.mlp[-1].weight.grad.abs().sum() > 0
    assert model.decoder.out.weight.grad.abs().sum() > 0


def test_trace_channels_last_keeps_the_output():
    from eviground.models.trace import TraceBranch

    torch.manual_seed(0)
    a = TraceBranch(pretrained=False, d=32).eval()
    b = TraceBranch(pretrained=False, d=32, channels_last=True).eval()
    b.load_state_dict(a.state_dict())
    assert b.proj.weight.is_contiguous(memory_format=torch.channels_last)
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    for k in ("s4", "s8", "s16", "t16"):
        assert torch.allclose(ya[k], yb[k], atol=1e-5)


def test_param_groups_use_the_two_learning_rates(model):
    groups = model.param_groups(1e-4, 5e-5, 0.05)
    assert [g["lr"] for g in groups] == [5e-5, 1e-4]
    n_lora = sum(p.numel() for p in groups[0]["params"])
    assert n_lora == sum(p.numel() for p in model.semantic.lora_parameters())
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_disable_dropout_makes_train_mode_deterministic():
    torch.manual_seed(1)
    m = build_model(load_config("debug").model)
    disable_dropout(m)
    m.eval()  # batch norm statistics fixed; dropout already 0
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        assert torch.equal(m(x)["logits"], m(x)["logits"])
    assert all(mod.p == 0.0 for mod in m.modules() if isinstance(mod, torch.nn.Dropout))


# --- losses ---------------------------------------------------------------------------------


def test_dice_is_zero_for_identical_masks():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    logits = (target * 2 - 1) * 50.0
    assert float(dice_loss(logits, target).max()) < 1e-6


def test_real_images_feed_bce_but_not_dice():
    """Decision of 15 Nov 2026: BCE on all samples, Dice on edited samples only."""
    mask_logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
    out = {
        "logits": torch.zeros(2, 3),
        "mask_logits": mask_logits,
        "alpha": torch.full((2, 256), 1 / 256),
    }
    labels = torch.tensor([0, 1])  # real and fully generated: all-zero masks
    loss = eviground_loss(out, labels, torch.zeros(2, 1, 32, 32))
    assert float(loss["dice"]) == 0.0
    assert float(loss["att"]) == 0.0
    assert float(loss["bce"]) > 0
    loss["total"].backward()
    assert float(mask_logits.grad.abs().sum()) > 0  # the real image's mask gets a BCE gradient


def test_attention_loss_is_zero_when_alpha_lies_inside_the_mask():
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, :64, :64] = 1  # the first 4 x 4 tokens of the 16 x 16 grid
    inside = torch.zeros(16, 16)
    inside[:4, :4] = 1 / 16
    assert abs(float(attention_loss(inside.flatten()[None], mask))) < 1e-6
    outside = torch.zeros(16, 16)
    outside[8:, 8:] = 1 / 64
    assert abs(float(attention_loss(outside.flatten()[None], mask)) - 1.0) < 1e-6


def test_loss_terms_and_lambdas():
    torch.manual_seed(0)
    out = {
        "logits": torch.randn(3, 3),
        "mask_logits": torch.randn(3, 1, 64, 64),
        "alpha": torch.softmax(torch.randn(3, 256), dim=1),
    }
    labels = torch.tensor([0, 1, EDITED])
    masks = torch.zeros(3, 1, 64, 64)
    masks[2, :, 8:40, 8:40] = 1
    base = eviground_loss(out, labels, masks, lambda_seg=1.0, lambda_att=0.1)
    no_att = eviground_loss(out, labels, masks, lambda_seg=1.0, lambda_att=0.0)
    expected = base["ce"] + base["bce"] + base["dice"] + 0.1 * base["att"]
    assert torch.allclose(base["total"], expected)
    assert torch.allclose(no_att["total"], base["total"] - 0.1 * base["att"])
    assert torch.allclose(base["ce"], F.cross_entropy(out["logits"], labels, label_smoothing=0.05))


# --- tiles and tiled prediction -------------------------------------------------------------


def test_tile_rule_full_coverage():
    assert tiles.tile_positions(256) == [0]
    assert tiles.tile_positions(200) == [0]
    assert tiles.tile_positions(600) == [0, 256, 344]
    assert tiles.tile_positions(1024) == [0, 256, 512, 768]  # full coverage (16 Nov 2026)
    assert tiles.tile_positions(1000) == [0, 256, 512, 744]  # last tile aligned to the border
    assert tiles.tile_count(1024, 1024) == 16
    assert tiles.tile_count(128, 128) == 1
    assert len(tiles.tile_boxes(700, 700)) == 9
    assert tiles.tile_boxes(180, 300) == [(0, 0, 256, 180), (44, 0, 300, 180)]
    for n in (128, 256, 300, 700, 1000, 1024, 2304):
        assert tiles.coverage(n) == 1.0
    for h, w in ((256, 256), (300, 700), (768, 512), (199, 640), (1024, 1024), (1003, 1003)):
        boxes = tiles.tile_boxes(h, w)
        assert len(boxes) == tiles.tile_count(h, w)
        assert all(r - le <= 256 and b - t <= 256 for le, t, r, b in boxes)  # never padded
        covered = torch.zeros(h, w, dtype=torch.bool)
        for le, t, r, b in boxes:
            covered[t:b, le:r] = True
        assert bool(covered.all())


def test_earlier_capped_rule_is_still_available():
    assert tiles.tile_positions(1024, max_per_axis=3) == [0, 384, 768]  # not covering
    assert tiles.coverage(1024, max_per_axis=3) == 0.75


def test_stitch_takes_the_mean_where_tiles_overlap():
    boxes = tiles.tile_boxes(256, 300)  # x = 0 and 44: columns 44-255 are covered twice
    maps = torch.stack([torch.zeros(256, 256), torch.ones(256, 256)])
    out = tiles.stitch(maps, boxes, 256, 300)
    assert float(out[0, 10]) == 0.0
    assert float(out[0, 100]) == 0.5
    assert float(out[0, 290]) == 1.0


def test_stitch_of_identity_tiles_gives_the_image_back():
    img = torch.rand(1003, 1003)
    boxes = tiles.tile_boxes(1003, 1003)
    maps = torch.stack([img[t:b, le:r] for le, t, r, b in boxes])
    assert torch.allclose(tiles.stitch(maps, boxes, 1003, 1003), img)


def test_stitch_fills_uncovered_pixels_from_the_nearest_tile():
    boxes = tiles.tile_boxes(1024, 256, max_per_axis=3)
    maps = torch.stack([torch.full((256, 256), float(i)) for i in range(len(boxes))])
    out = tiles.stitch(maps, boxes, 1024, 256)
    assert out.shape == (1024, 256)
    # tiles cover rows 0-255, 384-639, 768-1023; rows 256-383 and 640-767 take the nearest row
    assert float(out[300, 0]) == 0.0  # 45 rows from tile 0, 84 from tile 1
    assert float(out[360, 0]) == 1.0  # 24 rows from tile 1
    assert float(out[700, 0]) == 1.0
    assert float(out[1000, 0]) == 2.0


def test_tiled_prediction_with_one_edited_tile(monkeypatch, model):
    """An image whose top-left tile is called edited -> class edited, mask of the image size."""
    image = torch.rand(3, 300, 520)

    def fake_forward(x):
        n, _, h, w = x.shape
        logits = torch.tensor([[4.0, 0.0, -4.0]]).repeat(n, 1)
        logits[0] = torch.tensor([-4.0, -4.0, 4.0])  # first tile: edited
        mask = torch.full((n, 1, h, w), -10.0)
        mask[0] = 10.0
        return {"logits": logits, "mask_logits": mask}

    monkeypatch.setattr(model, "forward", fake_forward)
    res = model.predict_tiled(image)
    assert res["label"] == "edited"
    assert res["mask"].shape == (300, 520)
    assert res["tiles"] == 6
    assert float(res["mask"][0, 0]) > 0.99
    assert float(res["mask"][-1, -1]) < 0.01


def test_tiled_prediction_runs_fixed_size_batches(monkeypatch, model):
    image = torch.rand(3, 1024, 1024)
    sizes = []

    def fake_forward(x):
        sizes.append(x.shape[0])
        n, _, h, w = x.shape
        return {"logits": torch.zeros(n, 3), "mask_logits": torch.zeros(n, 1, h, w)}

    monkeypatch.setattr(model, "forward", fake_forward)
    res = model.predict_tiled(image, batch_size=5)
    assert res["tiles"] == 16 and res["coverage"] == 1.0
    assert sizes == [5, 5, 5, 1]
