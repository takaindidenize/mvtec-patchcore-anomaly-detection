"""Tests for the latency benchmark helpers (T10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from src.benchmark import measure, render_table
from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _counting_app() -> tuple[FastAPI, list[int]]:
    app = FastAPI()
    calls: list[int] = []

    @app.post("/predict")
    def predict(file: UploadFile) -> dict[str, float]:
        calls.append(1)
        return {"latency_ms": 2.0}

    return app, calls


def test_measure_sends_the_requested_requests_and_reports_percentiles() -> None:
    app, calls = _counting_app()
    with TestClient(app) as client:
        result = measure(client, [b"a", b"b"], requests=120, warmup=3)

    assert len(calls) == 1 + 3 + 120  # first request, warm-up, measured
    assert result["requests"] == 120
    assert result["server_p50_ms"] == result["server_p95_ms"] == 2.0
    assert 0 < result["client_p50_ms"] <= result["client_p95_ms"]


def test_table_lists_every_model_with_p50_and_p95() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "bottle.yaml")
    row = {
        "requests": 200.0,
        "first_request_ms": 5.0,
        "client_p50_ms": 4.0,
        "client_p95_ms": 6.0,
        "server_p50_ms": 3.0,
        "server_p95_ms": 5.0,
    }
    table = render_table({"patchcore": row, "autoencoder": row}, config)

    assert "| patchcore |" in table and "| autoencoder |" in table
    assert "p50" in table and "p95" in table
    assert "200 single-image" in table
