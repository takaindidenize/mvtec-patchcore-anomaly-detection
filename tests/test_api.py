"""Tests for the REST service (T9): every endpoint and every error status.

A fabricated bundle and a stand-in model replace the real artifact, so these run in
CI without weights or the dataset.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main
from src.api.main import MAX_UPLOAD_BYTES, create_app
from src.bundle import SCHEMA_VERSION, write_bundle

CROP = 32
THRESHOLD = 1.0


class _BrightnessModel:
    """Score = mean of the normalized image: white is anomalous, black is normal."""

    name = "fake"

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        maps = images.mean(dim=1).numpy().astype(np.float32)
        return images.mean(dim=(1, 2, 3)).numpy().astype(np.float32), maps


def _bundle(**overrides: Any) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "category": "widget",
        "model_name": "fake",
        "backbone": "none",
        "resolved_out_indices": [],
        "preprocessing": {
            "resize": 40,
            "center_crop": CROP,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "feature_map_size": (4, 4),
        "memory_bank": np.zeros((3, 4), dtype=np.float32),
        "coreset_ratio": 0.01,
        "gaussian_sigma": 4,
        "threshold": THRESHOLD,
        "target_fpr": 0.01,
        "seed": 7,
        "created_at": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "model_state": {"weights": np.zeros(3)},
    }
    bundle.update(overrides)
    return bundle


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace model construction and count how many times it happens."""
    calls: list[int] = []

    def fake_loader(bundle: dict[str, Any]) -> _BrightnessModel:
        calls.append(1)
        return _BrightnessModel()

    monkeypatch.setattr(main, "model_from_bundle", fake_loader)
    return calls


@pytest.fixture
def client(tmp_path: Path, loads: list[int]):
    path = tmp_path / "widget_fake_seed7.joblib"
    write_bundle(path, _bundle())
    with TestClient(create_app(path)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path: Path, loads: list[int]):
    with TestClient(create_app(tmp_path / "missing.joblib")) as test_client:
        yield test_client


def _image_bytes(color: int, fmt: str = "PNG", size: tuple[int, int] = (50, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buffer, format=fmt)
    return buffer.getvalue()


def _post(client: TestClient, data: bytes, content_type: str = "image/png"):
    return client.post("/predict", files={"file": ("upload", data, content_type)})


# --- /predict: success -----------------------------------------------------------


def test_predict_flags_the_anomalous_image(client: TestClient) -> None:
    white = _post(client, _image_bytes(255))
    black = _post(client, _image_bytes(0))

    assert white.status_code == 200 and black.status_code == 200
    assert white.json()["is_anomalous"] is True
    assert black.json()["is_anomalous"] is False
    assert white.json()["score"] > white.json()["threshold"] == THRESHOLD > black.json()["score"]


def test_predict_response_fields_and_heatmap_size_come_from_the_bundle(client: TestClient) -> None:
    body = _post(client, _image_bytes(255)).json()

    assert set(body) == {
        "score",
        "threshold",
        "is_anomalous",
        "heatmap_png_base64",
        "latency_ms",
        "model_version",
    }
    assert body["model_version"] == "widget_fake_seed7"
    assert body["latency_ms"] > 0
    heatmap = Image.open(io.BytesIO(base64.b64decode(body["heatmap_png_base64"])))
    assert heatmap.format == "PNG"
    # 32, not 224: the crop size is read from the bundle, never hardcoded.
    assert heatmap.size == (CROP, CROP)


def test_predict_accepts_jpeg(client: TestClient) -> None:
    response = _post(client, _image_bytes(255, "JPEG"), "image/jpeg")
    assert response.status_code == 200


def test_model_is_loaded_once_not_per_request(client: TestClient, loads: list[int]) -> None:
    _post(client, _image_bytes(255))
    _post(client, _image_bytes(0))
    assert len(loads) == 1


# --- /predict: errors ------------------------------------------------------------


def test_predict_413_when_file_too_large(client: TestClient) -> None:
    response = _post(client, b"x" * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 413
    assert response.json() == {"detail": "File too large"}


def test_predict_422_on_unsupported_content_type(client: TestClient) -> None:
    response = _post(client, b"hello", "text/plain")
    assert response.status_code == 422
    assert "Unsupported content type" in response.json()["detail"]


def test_predict_422_on_undecodable_image(client: TestClient) -> None:
    response = _post(client, b"not an image at all")
    assert response.status_code == 422
    assert "Cannot decode image" in response.json()["detail"]


def test_predict_422_when_content_does_not_match_declared_type(client: TestClient) -> None:
    response = _post(client, _image_bytes(255, "GIF"))
    assert response.status_code == 422


def test_predict_422_on_truncated_image(client: TestClient) -> None:
    response = _post(client, _image_bytes(255)[:40])
    assert response.status_code == 422


def test_predict_422_when_the_file_field_is_missing(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_predict_503_when_model_not_loaded(empty_client: TestClient) -> None:
    response = _post(empty_client, _image_bytes(255))
    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}


# --- /health and /info -----------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "category": "widget",
        "model_version": "widget_fake_seed7",
    }


def test_health_503_when_artifact_missing(empty_client: TestClient) -> None:
    response = empty_client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_health_503_when_artifact_is_corrupt(tmp_path: Path, loads: list[int]) -> None:
    path = tmp_path / "broken.joblib"
    path.write_bytes(b"garbage")
    with TestClient(create_app(path)) as test_client:
        assert test_client.get("/health").status_code == 503


def test_info_excludes_the_memory_bank_and_model_state(client: TestClient) -> None:
    response = client.get("/info")
    body = response.json()

    assert response.status_code == 200
    assert "memory_bank" not in body and "model_state" not in body
    assert body["category"] == "widget"
    assert body["threshold"] == THRESHOLD
    assert body["feature_map_size"] == [4, 4]
    assert body["preprocessing"]["center_crop"] == CROP


def test_info_503_when_model_not_loaded(empty_client: TestClient) -> None:
    response = empty_client.get("/info")
    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}
