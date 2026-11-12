"""Image facts: format sniffing, JPEG quality and subsampling, blockiness."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from eviground.data.imageinfo import blockiness, header_info, sniff_format


def _photo(size: int = 128, seed: int = 0) -> np.ndarray:
    """Smooth content plus mild noise: JPEG at low quality leaves visible blocks on it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 128 + 60 * np.sin(6 * xx + 3 * yy)[..., None] * np.array([1.0, 0.8, 0.6])
    return np.clip(base + rng.normal(0, 3, (size, size, 3)), 0, 255).astype(np.uint8)


def _encode(arr: np.ndarray, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp"), ("BMP", "bmp"), ("TIFF", "tiff")],
)
def test_sniff_format_uses_bytes_not_extension(fmt: str, expected: str) -> None:
    assert sniff_format(_encode(_photo(32), fmt)[:16]) == expected


def test_sniff_unknown() -> None:
    assert sniff_format(b"hello world, not an image") == "unknown"


@pytest.mark.parametrize("quality", [30, 75, 90, 95, 96, 100])
def test_jpeg_quality_exact_for_libjpeg_files(quality: int) -> None:
    info = header_info(_encode(_photo(), "JPEG", quality=quality))
    assert info["format"] == "jpeg"
    assert info["jpeg_quality"] == quality
    assert info["jpeg_quality_exact"] is True


def test_jpeg_quality_custom_tables_not_exact() -> None:
    custom = {0: [3] * 64, 1: [5] * 64}
    info = header_info(_encode(_photo(), "JPEG", qtables=custom))
    assert info["jpeg_quality_exact"] is False
    assert 1 <= info["jpeg_quality"] <= 100


@pytest.mark.parametrize(("sub", "label"), [(0, "4:4:4"), (1, "4:2:2"), (2, "4:2:0")])
def test_chroma_subsampling(sub: int, label: str) -> None:
    info = header_info(_encode(_photo(), "JPEG", quality=90, subsampling=sub))
    assert info["subsampling"] == label


def test_png_header() -> None:
    info = header_info(_encode(_photo(64), "PNG"))
    assert info["format"] == "png"
    assert (info["width"], info["height"]) == (64, 64)
    assert info["jpeg_quality"] is None and info["subsampling"] is None


def test_header_info_reads_paths_and_reports_errors(tmp_path) -> None:
    good = tmp_path / "a.weird_ext"
    good.write_bytes(_encode(_photo(40), "JPEG", quality=80))
    assert header_info(good)["jpeg_quality"] == 80  # format comes from bytes, not the name
    bad = tmp_path / "b.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    assert "error" in header_info(bad)
    assert "error" in header_info(tmp_path / "missing.png")


def _decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def test_blockiness_separates_compression_history() -> None:
    arr = _photo(256)
    lossless = blockiness(_decode(_encode(arr, "PNG")))
    heavy = blockiness(_decode(_encode(arr, "JPEG", quality=20)))
    assert abs(lossless - 1.0) < 0.1
    assert heavy > lossless + 0.2


def test_blockiness_phase() -> None:
    arr = _decode(_encode(_photo(256), "JPEG", quality=20))
    aligned = blockiness(arr)
    shifted = arr[3:, 5:]  # grid now starts at row phase 5, column phase 3
    assert blockiness(shifted, phase=(5, 3)) == pytest.approx(aligned, rel=0.1)
    assert blockiness(shifted, phase=(0, 0)) < aligned - 0.2
    assert blockiness(shifted, phase=None) == pytest.approx(aligned, rel=0.1)


def test_blockiness_rejects_tiny_images() -> None:
    with pytest.raises(ValueError):
        blockiness(np.zeros((10, 10), dtype=np.uint8))
