"""Coreset ratio sweep (T7).

Fits PatchCore at each ratio and records memory-bank size, fit time and single-image
query latency. Every measurement uses `train_fit` and `val_normal` only; the test
split is never opened, so the choice of ratio cannot be tuned against test metrics.

The `val_normal` median score is a normal-only coverage indicator: held-out normal
images sit closer to the bank when it covers normal appearance well. It says nothing
about how well defects are detected, which is why it is reported next to cost and is
not used on its own to pick a ratio.
"""

from __future__ import annotations

import argparse
import logging
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from src.config import Config, load_config
from src.data.mvtec import build_dataloaders
from src.models.backbone import build_backbone, resolve_out_indices
from src.models.patchcore import FEATURE_STRIDES, PatchCoreModel

LOGGER = logging.getLogger(__name__)

RATIOS = (0.1, 0.01, 0.001)


@dataclass(frozen=True)
class SweepRow:
    coreset_ratio: float
    bank_patches: int
    memory_bank_mb: float
    fit_seconds: float
    query_p50_ms: float
    query_p95_ms: float
    val_median_score: float


def measure(config: Config, ratio: float, backbone: torch.nn.Module, out_indices: list[int]):
    """Fit one ratio and time single-image queries on `val_normal`."""
    config = replace(config, model=replace(config.model, coreset_ratio=ratio))
    torch.manual_seed(config.seed)
    loaders = build_dataloaders(config)
    model = PatchCoreModel(
        config.model.backbone,
        out_indices,
        config.preprocessing.center_crop,
        ratio,
        config.model.gaussian_sigma,
        config.seed,
        module=backbone,
    )

    start = time.perf_counter()
    model.fit(loaders.train_fit)
    fit_seconds = time.perf_counter() - start

    images = [image[None] for batch in loaders.val_normal for image in batch["image"]]
    model.predict(images[0])  # warm-up, excluded from the timings
    latencies, scores = [], []
    for image in images:
        start = time.perf_counter()
        score, _ = model.predict(image)
        latencies.append((time.perf_counter() - start) * 1000.0)
        scores.append(float(score[0]))

    assert model.memory_bank is not None
    return SweepRow(
        coreset_ratio=ratio,
        bank_patches=int(model.memory_bank.shape[0]),
        memory_bank_mb=model.memory_bank_mb,
        fit_seconds=fit_seconds,
        query_p50_ms=float(np.percentile(latencies, 50)),
        query_p95_ms=float(np.percentile(latencies, 95)),
        val_median_score=float(np.median(scores)),
    ), len(images)


def render_table(rows: list[SweepRow], config: Config, val_images: int) -> str:
    lines = [
        "# Coreset ratio sweep",
        "",
        "Written by `src/coreset_sweep.py`; do not edit by hand.",
        "",
        f"- category: `{config.category}`, seed: `{config.seed}`, "
        f"backbone: `{config.model.backbone}`",
        f"- query latency: single-image `predict` over {val_images} `val_normal` images, "
        f"{torch.get_num_threads()} CPU threads, {platform.processor() or platform.machine()}",
        "- no test image was read",
        "",
        "| coreset_ratio | bank patches | memory_bank_mb | fit s | query p50 ms | query p95 ms "
        "| val_normal median score |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        marker = " (config)" if row.coreset_ratio == config.model.coreset_ratio else ""
        lines.append(
            f"| {row.coreset_ratio}{marker} | {row.bank_patches} | {row.memory_bank_mb:.2f} "
            f"| {row.fit_seconds:.1f} | {row.query_p50_ms:.1f} | {row.query_p95_ms:.1f} "
            f"| {row.val_median_score:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config, seed=args.seed)

    out_indices, _ = resolve_out_indices(
        config.model.backbone, FEATURE_STRIDES, config.preprocessing.center_crop
    )
    backbone = build_backbone(config.model.backbone, out_indices)

    rows, val_images = [], 0
    for ratio in RATIOS:
        row, val_images = measure(config, ratio, backbone, out_indices)
        LOGGER.info("ratio %s done: %s", ratio, row)
        rows.append(row)

    output = Path(config.output.results_dir) / "coreset.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(rows, config, val_images), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
