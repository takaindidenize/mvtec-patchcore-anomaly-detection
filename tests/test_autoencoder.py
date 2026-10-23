"""Tests for the convolutional autoencoder baseline (T5).

Run with two epochs on random tensors, so they need neither the dataset nor weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.autoencoder import AutoencoderModel

SIZE = 224


def _loader(count: int = 8, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 3, SIZE, SIZE, generator=generator) * 0.5
    return [{"image": images[i : i + 4]} for i in range(0, count, 4)]


def _fitted(seed: int = 0) -> AutoencoderModel:
    torch.manual_seed(seed)
    model = AutoencoderModel(SIZE, epochs=2)
    model.fit(_loader())
    return model


def test_predict_shapes_dtypes_and_score_is_map_max() -> None:
    model = _fitted()
    images = torch.randn(3, 3, SIZE, SIZE)
    scores, maps = model.predict(images)

    assert scores.shape == (3,) and scores.dtype == np.float32
    assert maps.shape == (3, SIZE, SIZE) and maps.dtype == np.float32
    assert np.all(maps >= 0)
    np.testing.assert_array_equal(scores, maps.reshape(3, -1).max(axis=1))


def test_training_reduces_reconstruction_error() -> None:
    torch.manual_seed(0)
    loader = _loader()
    untrained = AutoencoderModel(SIZE, epochs=0)
    untrained.fitted = True
    trained = AutoencoderModel(SIZE, epochs=10)
    trained.fit(loader)

    images = loader[0]["image"]
    assert trained.predict(images)[1].mean() < untrained.predict(images)[1].mean()


def test_same_seed_gives_identical_output() -> None:
    images = torch.randn(2, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(9))
    first = _fitted(seed=3).predict(images)
    second = _fitted(seed=3).predict(images)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_state_roundtrip_gives_identical_output() -> None:
    model = _fitted()
    images = torch.randn(2, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(5))
    restored = AutoencoderModel.from_state(model.get_state())
    np.testing.assert_array_equal(model.predict(images)[1], restored.predict(images)[1])


def test_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError):
        AutoencoderModel(SIZE).predict(torch.zeros(1, 3, SIZE, SIZE))
