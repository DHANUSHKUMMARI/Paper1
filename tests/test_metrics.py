"""Per-source validation accuracy (decision of 14 Nov 2026)."""

from __future__ import annotations

import pytest

from eviground.eval import metrics


def _case():
    y_true = ["real", "real", "real", "real", "generated", "generated", "edited", "edited"]
    y_pred = ["real", "real", "edited", "real", "generated", "real", "edited", "real"]
    sources = [
        "coco",
        "coco",
        "genimage",
        "sidset",
        "genimage",
        "sidset",
        "diffseg30k",
        "diffseg30k",
    ]
    return y_true, y_pred, sources


def test_accuracy_per_source_and_class():
    res = metrics.per_source_accuracy(*_case())
    assert res["overall"] == {"n": 8, "accuracy": 0.625, "balanced_accuracy": 0.5833}
    by = res["by_source"]
    assert by["coco"] == {"n": 2, "accuracy": 1.0, "by_class": {"real": {"n": 2, "accuracy": 1.0}}}
    assert by["genimage"]["accuracy"] == 0.5
    assert by["genimage"]["by_class"] == {
        "real": {"n": 1, "accuracy": 0.0},
        "generated": {"n": 1, "accuracy": 1.0},
    }
    assert by["sidset"]["accuracy"] == 0.5
    assert by["diffseg30k"]["by_class"]["edited"] == {"n": 2, "accuracy": 0.5}


def test_a_source_missing_from_validation_is_reported_empty():
    res = metrics.per_source_accuracy(["real"], ["real"], ["genimage"])
    assert res["by_source"]["coco"] == {"n": 0, "accuracy": None, "by_class": {}}
    assert list(res["by_source"]) == ["genimage", "sidset", "coco", "diffseg30k"]
    other = metrics.per_source_accuracy(["real"], ["real"], ["chameleon"])
    assert list(other["by_source"])[-1] == "chameleon"  # unknown sources are kept, last


def test_metrics_row_has_a_fixed_column_order():
    row = metrics.metrics_row(3, metrics.per_source_accuracy(*_case()))
    assert list(row)[:7] == [
        "epoch",
        "val_acc",
        "val_bal_acc",
        "val_acc_genimage",
        "val_acc_sidset",
        "val_acc_coco",
        "val_acc_diffseg30k",
    ]
    assert row["val_acc_coco"] == 1.0
    assert row["val_acc_coco_real"] == 1.0
    assert row["val_acc_genimage_real"] == 0.0
    again = metrics.metrics_row(3, metrics.per_source_accuracy(*_case()))
    assert list(again) == list(row)


def test_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        metrics.per_source_accuracy(["real"], ["real", "real"], ["coco"])
