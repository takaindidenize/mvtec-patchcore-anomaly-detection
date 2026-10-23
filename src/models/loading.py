"""Rebuild a fitted model from its artifact bundle."""

from __future__ import annotations

from typing import Any

from src.models.autoencoder import AutoencoderModel
from src.models.baseline_mahalanobis import MahalanobisModel
from src.models.patchcore import PatchCoreModel


def model_from_bundle(bundle: dict[str, Any]) -> Any:
    name = bundle["model_name"]
    if name == "mahalanobis":
        return MahalanobisModel.from_state(bundle["model_state"])
    if name == "autoencoder":
        return AutoencoderModel.from_state(bundle["model_state"])
    if name == "patchcore":
        return PatchCoreModel.from_bundle(bundle)
    raise ValueError(f"unknown model_name '{name}' in bundle")
