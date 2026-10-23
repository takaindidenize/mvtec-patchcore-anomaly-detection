"""Fit a model on `train_fit`, calibrate its threshold on `val_normal`, write the bundle.

This path never iterates the test split: the test loader is built by
`build_dataloaders` but nothing here reads from it (AGENTS.md section 2, rule 1).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.bundle import SCHEMA_VERSION, bundle_path, current_git_commit, write_bundle
from src.calibrate import calibrate_threshold
from src.config import Config, load_config
from src.data.mvtec import build_dataloaders
from src.models.autoencoder import AutoencoderModel
from src.models.base import AnomalyModel
from src.models.baseline_mahalanobis import MahalanobisModel
from src.models.patchcore import PatchCoreModel

LOGGER = logging.getLogger(__name__)

#: Model name (the `model.name` config value) -> factory. New models register here.
MODEL_FACTORIES: dict[str, Callable[[Config], Any]] = {
    "autoencoder": AutoencoderModel.from_config,
    "mahalanobis": MahalanobisModel.from_config,
    "patchcore": PatchCoreModel.from_config,
}


def build_model(config: Config) -> AnomalyModel:
    try:
        factory = MODEL_FACTORIES[config.model.name]
    except KeyError:
        raise ValueError(
            f"unknown model '{config.model.name}'; implemented: {sorted(MODEL_FACTORIES)}"
        ) from None
    return factory(config)


def score_loader(model: AnomalyModel, loader: DataLoader) -> np.ndarray:
    """Image scores for every image in `loader`."""
    scores = [model.predict(batch["image"])[0] for batch in loader]
    return np.concatenate(scores, axis=0)


def make_bundle(config: Config, model: Any, val_scores: np.ndarray) -> dict[str, Any]:
    """Assemble the section 12 bundle, including the calibrated threshold."""
    preprocessing = asdict(config.preprocessing)
    preprocessing["mean"] = list(preprocessing["mean"])
    preprocessing["std"] = list(preprocessing["std"])
    return {
        "schema_version": SCHEMA_VERSION,
        "category": config.category,
        "model_name": model.name,
        "backbone": model.backbone_name,
        "resolved_out_indices": list(model.out_indices),
        "preprocessing": preprocessing,
        "feature_map_size": tuple(model.feature_map_size),
        "memory_bank": getattr(model, "memory_bank", None),
        "coreset_ratio": config.model.coreset_ratio,
        "gaussian_sigma": config.model.gaussian_sigma,
        "threshold": calibrate_threshold(val_scores, config.target_fpr),
        "target_fpr": config.target_fpr,
        "seed": config.seed,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": current_git_commit(),
        "model_state": model.get_state(),
    }


def train(config: Config, model: Any | None = None) -> Path:
    """Fit, calibrate and write `artifacts/<category>_<model>_seed<seed>.joblib`."""
    torch.manual_seed(config.seed)
    model = model if model is not None else build_model(config)
    loaders = build_dataloaders(
        config, shuffle_train_fit=getattr(model, "shuffle_train_fit", False)
    )

    model.fit(loaders.train_fit)
    val_scores = score_loader(model, loaders.val_normal)
    bundle = make_bundle(config, model, val_scores)

    path = bundle_path(config.output.artifacts_dir, config.category, model.name, config.seed)
    write_bundle(path, bundle)
    LOGGER.info("wrote %s (threshold=%.6f)", path, bundle["threshold"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    parser.add_argument("--model", default=None, help="overrides model.name in the config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config, seed=args.seed)
    if args.model is not None:
        config = replace(config, model=replace(config.model, name=args.model))
    print(train(config))


if __name__ == "__main__":
    main()
