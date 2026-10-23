"""Tests for PatchCore (T6).

A tiny two-scale stand-in backbone replaces `wide_resnet50_2`, so these run in CI
without weights or the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.models.patchcore import PatchCoreModel, greedy_kcenter

SIZE = 224


class _TwoScaleBackbone(nn.Module):
    """Returns a stride-8 and a stride-16 map, like `layer2` and `layer3`."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.shallow = nn.Parameter(torch.randn(6, 3, 8, 8, generator=generator), False)
        self.deep = nn.Parameter(torch.randn(8, 3, 16, 16, generator=generator), False)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [
            nn.functional.conv2d(images, self.shallow, stride=8),
            nn.functional.conv2d(images, self.deep, stride=16),
        ]


def _model(seed: int = 0, ratio: float = 0.05) -> PatchCoreModel:
    return PatchCoreModel(
        "tiny", [0, 1], SIZE, coreset_ratio=ratio, gaussian_sigma=4, seed=seed,
        module=_TwoScaleBackbone(),
    )  # fmt: skip


def _normal_images(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, SIZE, SIZE, generator=generator) * 0.1


def _loader(images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    return [{"image": images[i : i + 4]} for i in range(0, len(images), 4)]


def test_greedy_kcenter_is_seeded_unique_and_covers() -> None:
    features = torch.randn(500, 8, generator=torch.Generator().manual_seed(0))
    first = greedy_kcenter(features, 50, seed=3)
    second = greedy_kcenter(features, 50, seed=3)

    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 50
    # Asking for everything returns every point.
    assert sorted(greedy_kcenter(features, 500, seed=0).tolist()) == list(range(500))


def test_fit_embeds_each_batch_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    calls = 0
    original = model._embed

    def counting(images: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(images)

    monkeypatch.setattr(model, "_embed", counting)
    model.fit(_loader(_normal_images(8)))
    assert calls == 2  # 8 images in batches of 4


def test_memory_bank_size_dtype_and_map_grid() -> None:
    model = _model(ratio=0.05)
    model.fit(_loader(_normal_images(8)))

    patches = 8 * (SIZE // 8) ** 2
    assert model.memory_bank is not None
    assert model.memory_bank.dtype == np.float32
    assert model.memory_bank.shape == (int(0.05 * patches), 6 + 8)
    assert model.feature_map_size == (SIZE // 8, SIZE // 8)
    assert model.memory_bank_mb == model.memory_bank.nbytes / (1024 * 1024)


def test_predict_shapes_and_score_is_map_max() -> None:
    model = _model()
    model.fit(_loader(_normal_images(8)))
    scores, maps = model.predict(_normal_images(3, seed=1))

    assert scores.shape == (3,) and scores.dtype == np.float32
    assert maps.shape == (3, SIZE, SIZE) and maps.dtype == np.float32
    assert np.all(maps >= 0)
    np.testing.assert_array_equal(scores, maps.reshape(3, -1).max(axis=1))


def test_defect_raises_score_and_is_localized() -> None:
    model = _model(ratio=0.2)
    model.fit(_loader(_normal_images(8)))

    clean = _normal_images(1, seed=2)
    defective = clean.clone()
    defective[:, :, 96:128, 96:128] += 3.0

    clean_score, _ = model.predict(clean)
    defect_score, defect_map = model.predict(defective)
    assert defect_score[0] > clean_score[0]

    peak_row, peak_col = np.unravel_index(np.argmax(defect_map[0]), defect_map[0].shape)
    assert 80 <= peak_row <= 144 and 80 <= peak_col <= 144


def test_same_seed_gives_identical_output() -> None:
    images = _normal_images(8)
    probe = _normal_images(2, seed=5)
    first, second = _model(seed=7), _model(seed=7)
    first.fit(_loader(images))
    second.fit(_loader(images))

    np.testing.assert_array_equal(first.memory_bank, second.memory_bank)
    for left, right in zip(first.predict(probe), second.predict(probe), strict=True):
        np.testing.assert_array_equal(left, right)


def test_bundle_roundtrip_gives_identical_output() -> None:
    model = _model()
    model.fit(_loader(_normal_images(8)))
    bundle = {
        "backbone": "tiny",
        "resolved_out_indices": [0, 1],
        "preprocessing": {"center_crop": SIZE},
        "coreset_ratio": 0.05,
        "gaussian_sigma": 4,
        "seed": 0,
        "memory_bank": model.memory_bank,
        "feature_map_size": model.feature_map_size,
    }
    restored = PatchCoreModel.from_bundle(bundle, module=_TwoScaleBackbone())

    probe = _normal_images(2, seed=6)
    np.testing.assert_array_equal(model.predict(probe)[1], restored.predict(probe)[1])


def test_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError):
        _model().predict(torch.zeros(1, 3, SIZE, SIZE))
