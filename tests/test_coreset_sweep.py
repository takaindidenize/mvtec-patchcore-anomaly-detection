"""Tests for the coreset sweep report (T7)."""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.coreset_sweep import RATIOS, SweepRow, render_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_table_has_one_row_per_ratio_and_marks_the_configured_one() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bottle.yaml")
    rows = [SweepRow(ratio, 100, 1.5, 2.0, 3.0, 4.0, 5.0) for ratio in RATIOS]
    table = render_table(rows, config, val_images=42)

    data_rows = [line for line in table.splitlines() if line.startswith("| 0.")]
    assert len(data_rows) == len(RATIOS)
    assert sum("(config)" in line for line in data_rows) == 1
    assert f"| {config.model.coreset_ratio} (config) |" in table
    assert "no test image was read" in table
