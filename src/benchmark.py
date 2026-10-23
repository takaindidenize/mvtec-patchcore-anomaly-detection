"""Single-image latency benchmark (T10).

Every request goes through the real FastAPI app, so the numbers include multipart
parsing, decoding, preprocessing, the model and heatmap rendering. The client is
`TestClient`, which runs in-process: network time is not part of the measurement.

Two latencies are recorded per request: the round trip seen by the client, and the
`latency_ms` the service reports for its own processing. Request images come from
`val_normal`; latency does not depend on image content, and the test split stays
out of it.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.bundle import bundle_path
from src.config import Config, load_config
from src.data.mvtec import build_splits

MODELS = ("mahalanobis", "autoencoder", "patchcore")
REQUESTS = 200
WARMUP_REQUESTS = 5


def measure(
    client: TestClient, images: list[bytes], requests: int, warmup: int
) -> dict[str, float]:
    """Latency percentiles over `requests` single-image requests, cycling `images`."""

    def send(index: int) -> tuple[float, float]:
        data = images[index % len(images)]
        start = time.perf_counter()
        response = client.post("/predict", files={"file": ("image.png", data, "image/png")})
        round_trip_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        return round_trip_ms, float(response.json()["latency_ms"])

    first_request_ms, _ = send(0)
    for index in range(warmup):
        send(index)

    samples = np.array([send(index) for index in range(requests)])
    client_ms, server_ms = samples[:, 0], samples[:, 1]
    return {
        "requests": float(requests),
        "first_request_ms": first_request_ms,
        "client_p50_ms": float(np.percentile(client_ms, 50)),
        "client_p95_ms": float(np.percentile(client_ms, 95)),
        "server_p50_ms": float(np.percentile(server_ms, 50)),
        "server_p95_ms": float(np.percentile(server_ms, 95)),
    }


def render_table(results: dict[str, dict[str, float]], config: Config) -> str:
    requests = int(next(iter(results.values()))["requests"])
    lines = [
        "# Latency benchmark",
        "",
        "Written by `src/benchmark.py`; do not edit by hand.",
        "",
        f"- category: `{config.category}`, seed: `{config.seed}`",
        f"- {requests} single-image `POST /predict` requests per model after {WARMUP_REQUESTS} "
        "warm-up requests, sent one at a time",
        "- client = round trip seen by the caller (in-process `TestClient`, no network); "
        "server = the `latency_ms` the service reports for decode, preprocessing, "
        "model and heatmap",
        f"- {torch.get_num_threads()} CPU threads, {platform.processor() or platform.machine()}",
        "- `first request` is the first request after startup warm-up",
        "",
        "| model | client p50 ms | client p95 ms | server p50 ms | server p95 ms "
        "| first request ms |",
        "|---|---|---|---|---|---|",
    ]
    for model_name, row in results.items():
        lines.append(
            f"| {model_name} | {row['client_p50_ms']:.1f} | {row['client_p95_ms']:.1f} "
            f"| {row['server_p50_ms']:.1f} | {row['server_p95_ms']:.1f} "
            f"| {row['first_request_ms']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed)
    images = [path.image_path.read_bytes() for path in build_splits(config).val_normal]

    results: dict[str, dict[str, float]] = {}
    for model_name in MODELS:
        artifact = bundle_path(
            config.output.artifacts_dir, config.category, model_name, config.seed
        )
        with TestClient(create_app(artifact)) as client:
            results[model_name] = measure(client, images, REQUESTS, WARMUP_REQUESTS)
        print(model_name, results[model_name])

    output = Path(config.output.results_dir) / "latency.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(results, config), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
