"""The single test-split evaluation (T8).

Run exactly once, after every design decision is frozen (AGENTS.md section 2, rule 1).
Each bundle already holds its threshold, calibrated on `val_normal` at train time, so
nothing here chooses or tunes anything: it loads the 3 models x 3 seeds, scores the
untouched test split and writes `results/metrics.md`.

To keep the run single-shot the script refuses to overwrite an existing
`results/metrics.md`. Deleting that file is a deliberate act, not a default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from src.bundle import bundle_path, load_bundle
from src.config import Config, load_config
from src.data.mvtec import build_dataloaders
from src.evaluate import Metrics, evaluate
from src.models.loading import model_from_bundle

MODELS = ("mahalanobis", "autoencoder", "patchcore")
SEEDS = (0, 1, 2)

BYTES_PER_MB = 1024 * 1024


def score_test_split(model: Any, loader: DataLoader) -> tuple[np.ndarray, ...]:
    """Labels, masks, image scores and anomaly maps for every test image."""
    labels, masks, scores, maps = [], [], [], []
    for batch in loader:
        batch_scores, batch_maps = model.predict(batch["image"])
        labels.append(batch["label"].numpy())
        masks.append(batch["mask"].numpy())
        scores.append(batch_scores)
        maps.append(batch_maps)
    return tuple(np.concatenate(parts, axis=0) for parts in (labels, masks, scores, maps))


def evaluate_bundle(bundle: dict[str, Any], loader: DataLoader) -> tuple[Metrics, float | None]:
    """Test metrics at the bundle's calibrated threshold, plus its memory-bank size."""
    model = model_from_bundle(bundle)
    labels, masks, scores, maps = score_test_split(model, loader)
    metrics = evaluate(labels, scores, masks, maps, bundle["threshold"])
    bank = bundle["memory_bank"]
    return metrics, (None if bank is None else bank.nbytes / BYTES_PER_MB)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) across seeds."""
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


METRIC_COLUMNS = (
    ("image_auroc", "image_auroc"),
    ("pixel_auroc", "pixel_auroc"),
    ("au_pro", "au_pro"),
    ("recall_at_threshold", "recall @ calibrated thr."),
    ("fpr_at_threshold", "FPR @ calibrated thr."),
    ("f1_max", "f1_max (oracle)"),
)


def render_metrics(
    results: dict[str, list[Metrics]], banks: dict[str, float | None], config: Config
) -> str:
    seeds = ", ".join(str(seed) for seed in SEEDS)
    lines = [
        "# Test-split results",
        "",
        "Written by `src/run_eval.py`; do not edit by hand.",
        "",
        f"- category: `{config.category}`, seeds: {seeds}; cells are mean ± sample std (ddof=1)",
        f"- calibrated threshold: quantile 1 - {config.target_fpr} of `val_normal` scores, "
        "fixed before this run",
        "- `f1_max (oracle)` picks its threshold with test labels: an upper bound, "
        "not achieved performance",
        "- `mahalanobis` is image-level only; its anomaly map is its image score repeated, "
        "so its pixel metrics carry no localization",
        "- latency is measured separately in `results/latency.md`",
        "",
        "| model | " + " | ".join(label for _, label in METRIC_COLUMNS) + " | memory_bank_mb |",
        "|---|" + "---|" * (len(METRIC_COLUMNS) + 1),
    ]
    for model_name, runs in results.items():
        cells = []
        for key, _ in METRIC_COLUMNS:
            mean, std = mean_std([getattr(run, key) for run in runs])
            cells.append(f"{mean:.3f} ± {std:.3f}")
        bank = banks[model_name]
        cells.append("n/a" if bank is None else f"{bank:.2f}")
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(config: Config) -> Path:
    output = Path(config.output.results_dir) / "metrics.md"
    if output.exists():
        raise SystemExit(
            f"{output} already exists. The test split is opened exactly once; "
            "delete the file only if you mean to open it again."
        )

    # Section 6: the test split is fixed by MVTec, so it is identical for every seed.
    loader = build_dataloaders(config).test

    results: dict[str, list[Metrics]] = {}
    banks: dict[str, float | None] = {}
    for model_name in MODELS:
        results[model_name] = []
        for seed in SEEDS:
            path = bundle_path(config.output.artifacts_dir, config.category, model_name, seed)
            metrics, bank_mb = evaluate_bundle(load_bundle(path), loader)
            results[model_name].append(metrics)
            banks[model_name] = bank_mb
            print(f"{model_name} seed {seed}: {metrics}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_metrics(results, banks, config), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"wrote {run(load_config(args.config))}")


if __name__ == "__main__":
    main()
