"""Sanity-audit tests. They build a tiny category on disk, so no dataset is needed."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.sanity import build_category_report


def _png(path: Path, size: tuple[int, int] = (16, 16), value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    Image.fromarray(np.full((height, width), value, dtype=np.uint8)).save(path)


def _category(root: Path) -> Path:
    """A minimal category that satisfies the data contract."""
    category = root / "widget"
    _png(category / "train" / "good" / "000.png")
    _png(category / "train" / "good" / "001.png")
    _png(category / "test" / "good" / "000.png")
    _png(category / "test" / "scratch" / "000.png")
    _png(category / "ground_truth" / "scratch" / "000_mask.png", value=255)
    return category


def test_clean_category_has_no_findings(tmp_path: Path) -> None:
    report = build_category_report(_category(tmp_path))
    assert report.findings == []
    assert report.train_good == 2
    assert report.test_counts == {"good": 1, "scratch": 1}
    assert report.mask_counts["scratch"] == 1


def test_missing_mask_is_reported(tmp_path: Path) -> None:
    category = _category(tmp_path)
    (category / "ground_truth" / "scratch" / "000_mask.png").unlink()

    report = build_category_report(category)
    assert report.missing_masks == ["scratch/000.png"]


def test_orphan_mask_is_reported(tmp_path: Path) -> None:
    category = _category(tmp_path)
    _png(category / "ground_truth" / "scratch" / "999_mask.png", value=255)

    report = build_category_report(category)
    assert report.orphan_masks == ["scratch/999_mask.png"]


def test_size_mismatch_is_reported(tmp_path: Path) -> None:
    category = _category(tmp_path)
    _png(category / "ground_truth" / "scratch" / "000_mask.png", size=(32, 32), value=255)

    report = build_category_report(category)
    assert len(report.size_mismatches) == 1
    assert "scratch/000.png" in report.size_mismatches[0]


def test_non_binary_mask_is_reported(tmp_path: Path) -> None:
    category = _category(tmp_path)
    _png(category / "ground_truth" / "scratch" / "000_mask.png", value=128)

    report = build_category_report(category)
    assert len(report.non_binary_masks) == 1
    assert "128" in report.non_binary_masks[0]


def test_masks_for_good_images_are_reported(tmp_path: Path) -> None:
    category = _category(tmp_path)
    _png(category / "ground_truth" / "good" / "000_mask.png", value=255)

    report = build_category_report(category)
    assert any("expected none" in problem for problem in report.structure_problems)


def test_ground_truth_folder_without_test_counterpart_is_reported(tmp_path: Path) -> None:
    category = _category(tmp_path)
    _png(category / "ground_truth" / "dent" / "000_mask.png", value=255)

    report = build_category_report(category)
    assert any("no test/ counterpart" in problem for problem in report.structure_problems)
