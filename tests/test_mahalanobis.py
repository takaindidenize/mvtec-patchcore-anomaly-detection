"""Tests for the Mahalanobis baseline, the bundle and the training path (T4).

A tiny stand-in backbone replaces the pretrained network, so these run in CI without
downloading weights or the dataset.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.bundle import BUNDLE_KEYS, bundle_path, load_bundle
from src.config import load_config
from src.models.baseline_mahalanobis import MahalanobisModel
from src.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIZE = 224


class _TinyBackbone(nn.Module):
    """Returns one 8-channel feature map, like a `features_only` network."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.weight = nn.Parameter(torch.randn(8, 3, 32, 32, generator=generator), False)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [nn.functional.conv2d(images, self.weight, stride=32)]


def _model() -> MahalanobisModel:
    return MahalanobisModel("tiny", [0], (7, 7), SIZE, module=_TinyBackbone())


def _loader(images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    return [{"image": images[i : i + 8]} for i in range(0, len(images), 8)]


def test_outlier_scores_higher_than_inliers() -> None:
    generator = torch.Generator().manual_seed(0)
    normal = torch.randn(64, 3, SIZE, SIZE, generator=generator) * 0.2
    model = _model()
    model.fit(_loader(normal))

    inlier_scores, _ = model.predict(torch.randn(8, 3, SIZE, SIZE, generator=generator) * 0.2)
    outlier_scores, _ = model.predict(torch.randn(8, 3, SIZE, SIZE, generator=generator) * 3.0)
    assert outlier_scores.min() > inlier_scores.max()


def test_predict_shapes_dtypes_and_uniform_map() -> None:
    generator = torch.Generator().manual_seed(1)
    model = _model()
    model.fit(_loader(torch.randn(32, 3, SIZE, SIZE, generator=generator)))

    scores, maps = model.predict(torch.randn(4, 3, SIZE, SIZE, generator=generator))
    assert scores.shape == (4,) and scores.dtype == np.float32
    assert maps.shape == (4, SIZE, SIZE) and maps.dtype == np.float32
    for score, anomaly_map in zip(scores, maps, strict=True):
        assert np.all(anomaly_map == score)


def test_state_roundtrip_gives_identical_scores(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(2)
    model = _model()
    model.fit(_loader(torch.randn(32, 3, SIZE, SIZE, generator=generator)))
    batch = torch.randn(4, 3, SIZE, SIZE, generator=generator)

    restored = MahalanobisModel.from_state(model.get_state(), module=_TinyBackbone())
    np.testing.assert_array_equal(model.predict(batch)[0], restored.predict(batch)[0])


def test_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError):
        _model().predict(torch.zeros(1, 3, SIZE, SIZE))


def _write_png(path: Path, *, valid: bool, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        rng = np.random.default_rng(seed)
        Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(path)
    else:
        path.write_bytes(b"not an image")


def test_train_writes_full_bundle_and_never_reads_test_images(tmp_path: Path) -> None:
    root = tmp_path / "mvtec" / "widget"
    for index in range(10):
        _write_png(root / "train" / "good" / f"{index:03d}.png", valid=True, seed=index)
    # Undecodable files: if training read any test image or mask it would raise.
    _write_png(root / "test" / "good" / "000.png", valid=False)
    _write_png(root / "test" / "crack" / "000.png", valid=False)
    _write_png(root / "ground_truth" / "crack" / "000_mask.png", valid=False)

    base = load_config(PROJECT_ROOT / "configs" / "bottle.yaml")
    config = replace(
        base,
        category="widget",
        data_root=str(tmp_path / "mvtec"),
        output=replace(base.output, artifacts_dir=str(tmp_path / "artifacts")),
        model=replace(base.model, name="mahalanobis"),
    )

    path = train(config, model=_model())

    assert path == bundle_path(tmp_path / "artifacts", "widget", "mahalanobis", config.seed)
    bundle = load_bundle(path)
    assert set(bundle) == BUNDLE_KEYS
    assert bundle["memory_bank"] is None
    assert bundle["preprocessing"]["center_crop"] == 224
    assert bundle["target_fpr"] == config.target_fpr
    assert np.isfinite(bundle["threshold"])
