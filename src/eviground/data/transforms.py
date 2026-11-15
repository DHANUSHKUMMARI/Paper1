"""Training augmentation after decoding (Milestone 3 plan, step 2).

Order: resize the whole image 0.5-1.5x (p 0.3; the short side stays >= the crop size; the mask
is resized with nearest neighbour) -> crop -> horizontal flip (p 0.5; image and mask) -> JPEG
re-encode of the crop, quality 60-100 (p 0.5) -> Gaussian blur, sigma 0-2 (p 0.2) -> Gaussian
noise, sigma 0-3/255 (p 0.2). The photometric steps touch the image only. Every random draw
comes from the generator the caller passes, so a (seed, epoch, draw) triple fixes the result.
``Augment(enabled=False)`` switches all of it off (overfit test, validation).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Augment:
    enabled: bool = True
    resize_p: float = 0.3
    resize_range: tuple[float, float] = (0.5, 1.5)
    flip_p: float = 0.5
    jpeg_p: float = 0.5
    jpeg_quality: tuple[int, int] = (60, 100)
    blur_p: float = 0.2
    blur_sigma: tuple[float, float] = (0.0, 2.0)
    noise_p: float = 0.2
    noise_sigma: tuple[float, float] = (0.0, 3.0 / 255.0)

    @classmethod
    def from_config(cls, cfg: Any | None) -> Augment:
        """From a config mapping (``data.augment``); missing keys keep their defaults."""
        if cfg is None:
            return cls()
        if isinstance(cfg, bool):
            return cls(enabled=cfg)
        kw = {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(cfg).items()}
        return cls(**kw)

    def scale(self, rng: np.random.Generator, width: int, height: int, crop: int) -> float:
        """Resize factor for the whole image (1.0 when not drawn); the short side stays at
        least ``crop``."""
        if not self.enabled or rng.random() >= self.resize_p:
            return 1.0
        lo = max(self.resize_range[0], crop / min(width, height))
        hi = self.resize_range[1]
        return 1.0 if lo >= hi else float(rng.uniform(lo, hi))

    def photometric(self, crop: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        """Flip, JPEG, blur and noise on an (H, W, 3) uint8 crop; returns the crop and whether
        it was flipped (the caller flips the mask)."""
        if not self.enabled:
            return crop, False
        flip = bool(rng.random() < self.flip_p)
        if flip:
            crop = crop[:, ::-1]
        img = Image.fromarray(np.ascontiguousarray(crop))
        if rng.random() < self.jpeg_p:
            q = int(rng.integers(self.jpeg_quality[0], self.jpeg_quality[1] + 1))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if rng.random() < self.blur_p:
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(*self.blur_sigma))))
        out = np.asarray(img, dtype=np.uint8)
        if rng.random() < self.noise_p:
            sigma = float(rng.uniform(*self.noise_sigma)) * 255.0
            noisy = out.astype(np.float32) + rng.normal(0.0, sigma, out.shape).astype(np.float32)
            out = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
        return out, flip


def is_identity(scale: float) -> bool:
    """True for the "no resize" factor 1.0 (returned as is by :meth:`Augment.scale`)."""
    return abs(scale - 1.0) < 1e-12


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_image(img: Image.Image, scale: float) -> Image.Image:
    if is_identity(scale):
        return img
    return img.resize(scaled_size(*img.size, scale), Image.Resampling.BICUBIC)


def resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask."""
    if is_identity(scale):
        return mask
    h, w = mask.shape
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(m.resize(scaled_size(w, h, scale), Image.Resampling.NEAREST)) > 127
