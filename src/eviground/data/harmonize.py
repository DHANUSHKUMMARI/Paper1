"""Format harmonization of the training data (Research Specification section 8, M1, rev 49).

Rule: within each source every class ends with the same compression history, set by the most
heavily compressed class of that source. Lossless or less compressed images are JPEG-compressed
once, as whole images before cropping; an image already at that compression is left alone.

| Source | Harmonization |
| --- | --- |
| GenImage | generated (PNG) -> JPEG with quality and subsampling sampled from the measured
  distribution of the real training JPEGs; real images untouched |
| SID-Set | real, synthetic and PNG tampered -> JPEG quality 75, 4:2:0; images already at 75 /
  4:2:0 untouched; JPEGs below quality 75 skipped and counted |
| DiffSeg30k, COCO negatives | all PNG after preparation -> JPEG with the GenImage real
  distribution; COCO negatives take sizes drawn from DiffSeg30k's measured sizes |

The plan for each image is a pure function of its header facts and its id (seed 42), so the
same image is always compressed the same way, in every session and every epoch.
"""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from eviground.data.imageinfo import LOSSLESS_FORMATS

SEED = 42
MIN_SIDE = 256  # training crops are 256x256; smaller images are skipped and counted
SUBSAMPLING_CODE = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
SID_TARGET = (75, "4:2:0")

# source -> rule (see the module docstring)
RULES = {
    "genimage": "real_distribution",
    "sidset": "sid_q75",
    "diffseg30k": "genimage_distribution",
    "coco": "genimage_distribution",
}


def _uid_int(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big")


@dataclass
class QualityDistribution:
    """Measured joint distribution of (JPEG quality, chroma subsampling)."""

    counts: dict[tuple[int, str], int] = field(default_factory=dict)

    @classmethod
    def measure(cls, infos: Iterable[Mapping[str, Any]]) -> QualityDistribution:
        """From header facts of JPEG files; greyscale and non-standard subsampling are left out
        (the images to compress are colour, so only colour settings can be sampled)."""
        counts: Counter[tuple[int, str]] = Counter()
        seen = 0
        for info in infos:
            seen += 1
            q, ss = info.get("jpeg_quality"), info.get("subsampling")
            if info.get("format") == "jpeg" and q is not None and ss in SUBSAMPLING_CODE:
                counts[(int(q), str(ss))] += 1
        if not counts:
            raise ValueError(
                f"no colour JPEG among the {seen} measured images (are the real training "
                "images staged and scanned?)"
            )
        return cls(dict(counts))

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    def _table(self) -> tuple[list[tuple[int, str]], np.ndarray]:
        keys = sorted(self.counts)
        p = np.array([self.counts[k] for k in keys], dtype=np.float64)
        return keys, p / p.sum()

    def sample(self, uid: str, seed: int = SEED) -> tuple[int, str]:
        """Deterministic draw for one image id."""
        keys, p = self._table()
        rng = np.random.default_rng([seed, _uid_int(uid)])
        return keys[int(rng.choice(len(keys), p=p))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "pairs": [[q, ss, c] for (q, ss), c in sorted(self.counts.items())],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> QualityDistribution:
        return cls({(int(q), str(ss)): int(c) for q, ss, c in d["pairs"]})

    def summary(self) -> dict[str, Any]:
        q: Counter[int] = Counter()
        ss: Counter[str] = Counter()
        for (qq, s), c in self.counts.items():
            q[qq] += c
            ss[s] += c
        top = q.most_common(1)[0]
        return {
            "n": self.n,
            "top_quality": top[0],
            "top_quality_share": round(top[1] / self.n, 4),
            "quality_hist": {str(k): v for k, v in sorted(q.items())},
            "subsampling": dict(ss),
        }


def plan_image(
    source: str,
    label: str,
    info: Mapping[str, Any],
    uid: str,
    dist: QualityDistribution | None,
    *,
    seed: int = SEED,
    min_side: int | None = MIN_SIDE,
) -> dict[str, Any]:
    """What harmonization does to one training image.

    Returns ``{"action": "keep" | "compress" | "skip", "quality", "subsampling", "reason"}``.
    ``info`` holds the header facts (format, width, height, jpeg_quality, subsampling).
    ``dist`` is the GenImage real distribution (needed by every rule except SID-Set).
    """
    fmt = info.get("format")
    if info.get("error") or fmt is None:
        return {"action": "skip", "quality": None, "subsampling": None, "reason": "unreadable"}
    w, h = info.get("width") or 0, info.get("height") or 0
    if min_side and min(w, h) < min_side:
        return {"action": "skip", "quality": None, "subsampling": None, "reason": "too_small"}
    rule = RULES[source]
    lossless = fmt in LOSSLESS_FORMATS

    def compress(q: int, ss: str, reason: str) -> dict[str, Any]:
        return {"action": "compress", "quality": int(q), "subsampling": ss, "reason": reason}

    keep = {
        "action": "keep",
        "quality": info.get("jpeg_quality"),
        "subsampling": info.get("subsampling"),
    }
    if rule == "sid_q75":
        tq, tss = SID_TARGET
        if lossless:
            return compress(tq, tss, "lossless")
        if fmt != "jpeg":
            return {"action": "skip", "quality": None, "subsampling": None, "reason": f"{fmt}"}
        q, ss = info.get("jpeg_quality"), info.get("subsampling")
        if q is None:
            return {"action": "skip", "quality": None, "subsampling": None, "reason": "no_q"}
        if q < tq:
            return {"action": "skip", "quality": q, "subsampling": ss, "reason": "below_q75"}
        if q == tq and ss in (tss, None):  # None: greyscale JPEG, no chroma to subsample
            return {**keep, "reason": "at_target"}
        return compress(tq, tss, "above_q75" if q > tq else "q75_other_subsampling")
    if dist is None:
        raise ValueError(f"{source}: the GenImage real quality distribution is required")
    if lossless:
        q, ss = dist.sample(uid, seed)
        return compress(q, ss, "lossless")
    if fmt == "jpeg":
        if rule == "real_distribution" and label != "real":
            return {**keep, "reason": "generated_already_jpeg"}
        return {**keep, "reason": "real_jpeg" if label == "real" else "already_jpeg"}
    return {"action": "skip", "quality": None, "subsampling": None, "reason": f"{fmt}"}


def encode_jpeg(img: Image.Image, quality: int, subsampling: str) -> bytes:
    """One JPEG compression (libjpeg via Pillow), colour, no metadata."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality), subsampling=SUBSAMPLING_CODE[subsampling])
    return buf.getvalue()


def apply_plan(data: bytes, plan: Mapping[str, Any]) -> bytes | None:
    """The harmonized file bytes (None for a skipped image)."""
    action = plan["action"]
    if action == "keep":
        return data
    if action == "skip":
        return None
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return encode_jpeg(img, plan["quality"], plan["subsampling"])


@dataclass
class SizeDistribution:
    """Measured image sizes (width, height) of a source, to draw sizes from."""

    counts: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def measure(cls, sizes: Iterable[tuple[int, int]]) -> SizeDistribution:
        counts: Counter[tuple[int, int]] = Counter((int(w), int(h)) for w, h in sizes)
        if not counts:
            raise ValueError("no image sizes to measure")
        return cls(dict(counts))

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    def sample(self, uid: str, seed: int = SEED) -> tuple[int, int]:
        """Deterministic draw for one image id (same id, same size, every run)."""
        keys = sorted(self.counts)
        p = np.array([self.counts[k] for k in keys], dtype=np.float64)
        rng = np.random.default_rng([seed, _uid_int("size:" + uid)])
        return keys[int(rng.choice(len(keys), p=p / p.sum()))]

    def summary(self, top: int = 10) -> dict[str, Any]:
        common = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return {
            "n": self.n,
            "distinct": len(self.counts),
            "top": {f"{w}x{h}": c for (w, h), c in common},
        }


def prepare_to_size(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """COCO real negatives at a size drawn from DiffSeg30k's measured sizes (decision of
    14 Nov 2026): resized (Lanczos) by the smallest factor that covers the target, then
    centre-cropped to it. An image already at the target size is not resampled."""
    img = img.convert("RGB")
    tw, th = size
    w, h = img.size
    if (w, h) == (tw, th):
        return img
    scale = max(tw / w, th / h)
    nw, nh = max(tw, round(w * scale)), max(th, round(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def resampling(src: tuple[int, int], target: tuple[int, int]) -> str:
    """'unchanged', 'cropped' (scale 1), 'downscaled' or 'upscaled' for :func:`prepare_to_size`."""
    if tuple(src) == tuple(target):
        return "unchanged"
    scale = max(target[0] / src[0], target[1] / src[1])
    if scale == 1:
        return "cropped"
    return "downscaled" if scale < 1 else "upscaled"


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def plan_summary(plans: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of action:reason."""
    c: Counter[str] = Counter(f"{p['action']}:{p['reason']}" for p in plans)
    return dict(sorted(c.items()))
