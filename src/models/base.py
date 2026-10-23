"""The interface every anomaly model implements (AGENTS.md section 8)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class AnomalyModel(Protocol):
    name: str

    def fit(self, loader: torch.utils.data.DataLoader) -> None:
        """Build the model from defect-free images only."""

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            images: (B, 3, 224, 224) float32, already preprocessed.
        Returns:
            image_scores: (B,) float32. Higher = more anomalous. Unbounded.
            anomaly_maps: (B, 224, 224) float32, same scale as image_scores.
        """

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> AnomalyModel: ...
