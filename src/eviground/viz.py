"""Figures drawn with seaborn from saved result files (Dhanush, 17 Nov 2026: seaborn for plots).

Every figure is drawn from what a notebook already wrote to Drive (its result JSON and per-image
CSV), never from objects held in memory. So a figure can be redrawn, or restyled, on a CPU runtime
without running a model again.

The ``*_rows`` functions turn results into tidy rows (one value per row) and need nothing beyond
the standard library; the ``*_figure(s)`` functions import seaborn, matplotlib and pandas when
called. Each figure is saved as PNG (for viewing) and PDF (for the paper).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from eviground.eval.locdiag import SIZE_BINS, size_bin

THEME = {"context": "notebook", "style": "whitegrid", "palette": "colorblind"}
DPI = 150

# training curves of the CP3 run (keys of metrics.csv), split into the two panels
MASK_CURVES = {
    "val_mask_f1": "validation mask F1",
    "val_mask_miou_binary": "validation two-class mIoU",
    "val_mask_iou_pooled": "validation edited IoU (DiffSeg30k mIoU)",
    "train_eval_mask_f1": "training subset mask F1",
    "train_eval_mask_miou_binary": "training subset two-class mIoU",
}
REAL_CURVES = {
    "val_real_any_pred": "real crops with any mark",
    "val_real_pred_area": "marked share of real crops",
}
PIXEL_SWEEP = {
    "miou_diffseg30k": "edited IoU (DiffSeg30k mIoU)",
    "f1_pooled": "pooled pixel F1",
    "mean_boundary_f1": "boundary F1 (per-image mean)",
}
IMAGE_SWEEP = {
    "edited_called_edited": "edited images called edited",
    "real_called_edited": "real images called edited",
}
PREDICTORS = {
    "model_iou": "EviGround",
    "probe_iou": "CLIP patch probe",
    "ceiling16_iou": "ceiling, 16×16 patches",
    "ceiling64_iou": "ceiling, 64×64 decoder grid",
}
HEAD_NAMES = {"model": "EviGround", "probe": "CLIP patch probe"}
P1_HEADS = {
    "detection": "EviGround (detection)",
    "detection_probe": "CLIP probe (detection)",
    "unified": "EviGround (unified)",
    "unified_probe": "CLIP probe (unified)",
}
SIZE_ORDER = [size_bin(lo) for lo, _hi in SIZE_BINS]


def _num(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Tidy rows (no plotting library needed)
# --------------------------------------------------------------------------------------


def curve_rows(curves: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """``{metric key: per-epoch values}`` -> rows ``epoch`` (from 1), ``panel``, ``metric``,
    ``value``; missing values are left out."""
    rows = []
    for panel, names in (("mask", MASK_CURVES), ("real", REAL_CURVES)):
        for key, label in names.items():
            for i, v in enumerate(curves.get(key) or []):
                x = _num(v)
                if x is not None:
                    rows.append({"epoch": i + 1, "panel": panel, "metric": label, "value": x})
    return rows


def sweep_rows(sweep: Mapping[str, Any], heads: Iterable[str]) -> list[dict[str, Any]]:
    """The ``threshold_sweep`` block -> rows ``kind`` (pixel / image), ``head``, ``threshold``,
    ``metric``, ``value``."""
    rows = []
    for head in heads:
        block = sweep.get(head) or {}
        for kind, names in (("pixel", PIXEL_SWEEP), ("image", IMAGE_SWEEP)):
            for r in block.get(kind) or []:
                rows += _sweep_point(r, kind, HEAD_NAMES.get(head, head), names)
    return rows


def _sweep_point(
    r: Mapping[str, Any], kind: str, head: str, names: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One sweep row (one threshold) -> one tidy row per metric it holds."""
    base = {"kind": kind, "head": head, "threshold": float(r["threshold"])}
    values = ((label, _num(r.get(key))) for key, label in names.items())
    return [{**base, "metric": label, "value": x} for label, x in values if x is not None]


def per_image_rows(edited_csv: Path) -> list[dict[str, Any]]:
    """The diagnosis' per-image CSV -> rows ``uid``, ``size_bin``, ``editor`` (one row per editor
    of a multi-turn image), ``base_type``, ``predictor``, ``iou``."""
    with open(edited_csv, newline="") as fh:
        return [row for r in csv.DictReader(fh) for row in _image_rows(r)]


def _image_rows(r: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One CSV row (one image) -> one tidy row per predictor with a value and per editor."""
    share = _num(r.get("share"))
    base = {
        "uid": r["uid"],
        "size_bin": "none" if share is None else size_bin(share),
        "base_type": r.get("base_type") or "unknown",
    }
    editors = [e for e in (r.get("editors") or "").split("|") if e] or ["unknown"]
    ious = ((name, _num(r.get(col))) for col, name in PREDICTORS.items())
    return [
        {**base, "editor": editor, "predictor": name, "iou": iou}
        for name, iou in ious
        if iou is not None
        for editor in editors
    ]


def p1_rows(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The CP4 P1 table (``cp4.p1_table``) -> rows ``head``, ``version``, ``generator``,
    ``acc``, ``ap``."""
    rows = []
    for head, versions in table.items():
        for version, t in versions.items():
            for gen, g in (t.get("per_generator") or {}).items():
                rows.append(
                    {
                        "head": P1_HEADS.get(head, head),
                        "version": version,
                        "generator": gen,
                        "acc": float(g["acc"]),
                        "ap": _num(g.get("ap")),
                    }
                )
    return rows


# --------------------------------------------------------------------------------------
# Figures (seaborn)
# --------------------------------------------------------------------------------------


def _libs() -> tuple[Any, Any, Any]:
    import matplotlib

    if not _interactive():  # scripts and tests: no display
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(**THEME)
    return sns, plt, pd


def _interactive() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _legend_below(sns: Any, a: Any, ncol: int = 2) -> None:
    """The legend under the axes, so it never covers data."""
    if a.get_legend() is not None:
        sns.move_legend(
            a,
            "upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=ncol,
            title=None,
            frameon=False,
            fontsize=9,
        )


def _save(fig: Any, plt: Any, out_dir: Path, name: str, show: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return png


def diagnosis_figures(
    result: Mapping[str, Any],
    edited_csv: Path,
    out_dir: Path,
    *,
    heads: Sequence[str] = ("model", "probe"),
    seed: int = 42,
    show: bool = True,
) -> list[Path]:
    """The localization diagnosis' figures from ``localization_diagnosis_seed<seed>.json``
    (``result``) and its per-image CSV: training curves, the validation threshold sweep, and
    per-image IoU by edit size, by editor and by base type. Returns the PNG paths."""
    sns, plt, pd = _libs()
    per = pd.DataFrame(per_image_rows(Path(edited_csv)))
    drawn = (
        _curves_figure(sns, plt, pd.DataFrame(curve_rows(result.get("curves") or {}))),
        _sweep_figure(
            sns, plt, pd.DataFrame(sweep_rows(result.get("threshold_sweep") or {}, heads))
        ),
        _size_figure(sns, plt, per),
        _editor_figure(sns, plt, per),
    )
    names = ("curves", "threshold_sweep", "iou_by_size", "iou_by_editor")
    return [
        _save(fig, plt, out_dir, f"diagnosis_{name}_seed{seed}", show)
        for name, fig in zip(names, drawn, strict=True)
        if fig is not None
    ]


def _curves_figure(sns: Any, plt: Any, curves: Any) -> Any:
    """CP3 training curves: mask metrics (with the G2 bar) and marks on real crops."""
    if curves.empty:
        return None
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for a, panel, title in (
        (ax[0], "mask", "Mask metrics on validation crops (CP3 training)"),
        (ax[1], "real", "Marks on real validation crops"),
    ):
        d = curves[curves.panel == panel]
        if not d.empty:
            sns.lineplot(data=d, x="epoch", y="value", hue="metric", marker="o", ax=a)
            _legend_below(sns, a)
        a.set_title(title)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].axhline(0.80, ls=":", c="k", lw=1)
    ax[0].text(1, 0.81, "G2 bar 0.80", fontsize=8)
    ax[1].set_yscale("symlog", linthresh=0.01)
    ax[1].set_ylim(bottom=0)
    return fig


def _sweep_figure(sns: Any, plt: Any, sweep: Any) -> Any:
    """Validation threshold sweep: pixel metrics and image calls, a dotted line at 0.5."""
    if sweep.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for a, kind, xlabel, title in (
        (ax[0], "pixel", "pixel threshold", "Validation pixel metrics vs threshold"),
        (ax[1], "image", "P(edited) threshold", "Validation image calls vs threshold"),
    ):
        d = sweep[sweep.kind == kind]
        if not d.empty:
            sns.lineplot(
                data=d, x="threshold", y="value", hue="metric", style="head", marker="o", ax=a
            )
            _legend_below(sns, a)
        a.axvline(0.5, ls=":", c="k", lw=1)
        a.set_xlabel(xlabel)
        a.set_title(title)
    return fig


def _size_figure(sns: Any, plt: Any, per: Any) -> Any:
    """Per-image IoU by edit size, beside the resolution ceilings, with n per bin."""
    if per.empty:
        return None
    one = per.drop_duplicates(["uid", "predictor"])  # editors are exploded: one row per image
    order = [b for b in SIZE_ORDER if b in set(one.size_bin)]
    fig, a = plt.subplots(figsize=(13, 4.5))
    sns.boxplot(
        data=one,
        x="size_bin",
        y="iou",
        hue="predictor",
        order=order,
        hue_order=[p for p in PREDICTORS.values() if p in set(one.predictor)],
        showfliers=False,
        ax=a,
    )
    counts = one[one.predictor == PREDICTORS["model_iou"]].size_bin.value_counts()
    a.set_xticks(range(len(order)), [f"{b}\n(n={counts.get(b, 0)})" for b in order])
    a.set_xlabel("edited share of the image")
    a.set_ylabel("per-image edited IoU")
    a.set_title(
        "Per-image IoU by edit size (validation, training editors), with the resolution ceilings"
    )
    _legend_below(sns, a, ncol=4)
    a.set_ylim(0, 1.02)
    return fig


def _editor_figure(sns: Any, plt: Any, per: Any) -> Any:
    """Per-image IoU of the model and the probe by editor and by base type."""
    if per.empty:
        return None
    heads_only = per[per.predictor.isin([PREDICTORS["model_iou"], PREDICTORS["probe_iou"]])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    sns.boxplot(data=heads_only, x="editor", y="iou", hue="predictor", showfliers=False, ax=ax[0])
    ax[0].set_title("Per-image IoU by editor (an image counts once per editor it went through)")
    ax[0].tick_params(axis="x", rotation=20)
    sns.boxplot(
        data=heads_only.drop_duplicates(["uid", "predictor"]),
        x="base_type",
        y="iou",
        hue="predictor",
        showfliers=False,
        ax=ax[1],
    )
    ax[1].set_title("By base type")
    ax[1].set_xlabel("base image")
    for a in ax:
        a.set_ylabel("per-image edited IoU")
        a.set_ylim(0, 1.02)
    ax[1].get_legend().remove()
    _legend_below(sns, ax[0])
    return fig


def p1_figure(
    table: Mapping[str, Any],
    out_dir: Path,
    *,
    bar: float = 0.721,
    seed: int = 42,
    show: bool = True,
) -> Path | None:
    """P1 accuracy per generator, one panel per version (raw / harmonized), one bar colour per
    model or probe, with the go/no-go bar (GenImage ResNet-50 average). Returns the PNG path."""
    sns, plt, pd = _libs()
    df = pd.DataFrame(p1_rows(table))
    if df.empty:
        return None
    versions = [v for v in ("raw", "harmonized") if v in set(df.version)]
    g = sns.catplot(
        data=df,
        x="generator",
        y="acc",
        hue="head",
        col="version",
        col_order=versions,
        kind="bar",
        height=4.2,
        aspect=1.6,
    )
    for a in g.axes.flat:
        a.axhline(bar, ls=":", c="k", lw=1)
        a.set_ylim(0, 1)
        a.tick_params(axis="x", rotation=30)
    g.set_axis_labels("generator", "accuracy (P1 development split)")
    g.set_titles("{col_name} files")
    sns.move_legend(
        g, "upper center", bbox_to_anchor=(0.45, -0.04), ncol=4, title=None, frameon=False
    )
    g.figure.suptitle(f"P1 accuracy per generator; dotted line: {bar}", y=1.03)
    return _save(g.figure, plt, out_dir, f"p1_dev_accuracy_seed{seed}", show)
