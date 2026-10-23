"""Tests for the single-shot test evaluation (T8), on synthetic data only."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src import run_eval
from src.config import load_config
from src.evaluate import Metrics
from src.run_eval import evaluate_bundle, mean_std, render_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIZE = 16


class _PerfectModel:
    """Scores an image by the brightness the loader encodes its label in."""

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        maps = images[:, 0].numpy().astype(np.float32)
        return maps.reshape(len(maps), -1).max(axis=1), maps


def _loader() -> list[dict[str, torch.Tensor]]:
    images, masks, labels = [], [], []
    for label in (0, 1, 0, 1):
        image = torch.zeros(3, SIZE, SIZE)
        mask = torch.zeros(SIZE, SIZE)
        if label:
            image[0, 4:8, 4:8] = 1.0
            mask[4:8, 4:8] = 1.0
        images.append(image)
        masks.append(mask)
        labels.append(label)
    return [
        {"image": torch.stack(images), "mask": torch.stack(masks), "label": torch.tensor(labels)}
    ]


def test_evaluate_bundle_uses_the_bundle_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "model_from_bundle", lambda bundle: _PerfectModel())
    bundle = {"threshold": 0.5, "memory_bank": np.zeros((4, 1024), dtype=np.float32)}

    metrics, bank_mb = evaluate_bundle(bundle, _loader())  # type: ignore[arg-type]

    assert metrics.image_auroc == 1.0
    assert metrics.recall_at_threshold == 1.0
    assert metrics.fpr_at_threshold == 0.0
    assert bank_mb == 4 * 1024 * 4 / (1024 * 1024)


def test_models_without_a_memory_bank_report_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "model_from_bundle", lambda bundle: _PerfectModel())
    _, bank_mb = evaluate_bundle({"threshold": 0.5, "memory_bank": None}, _loader())  # type: ignore[arg-type]
    assert bank_mb is None


def test_mean_std_is_the_sample_std() -> None:
    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0 and std == 1.0


def test_render_labels_the_oracle_and_marks_missing_banks() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bottle.yaml")
    run = Metrics(0.9, 0.8, 0.7, 0.95, 0.6, 0.05)
    table = render_metrics(
        {"mahalanobis": [run, run, run], "patchcore": [run, run, run]},
        {"mahalanobis": None, "patchcore": 7.67},
        config,
    )
    assert "f1_max (oracle)" in table
    assert "upper bound" in table
    assert "| mahalanobis |" in table and "n/a |" in table
    assert "0.900 ± 0.000" in table


def test_run_refuses_to_open_the_test_split_twice(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "bottle.yaml")
    config = replace(base, output=replace(base.output, results_dir=str(tmp_path)))
    (tmp_path / "metrics.md").write_text("earlier run", encoding="utf-8")

    with pytest.raises(SystemExit):
        run_eval.run(config)
    assert (tmp_path / "metrics.md").read_text(encoding="utf-8") == "earlier run"
