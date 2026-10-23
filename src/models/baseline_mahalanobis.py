"""Global Mahalanobis baseline.

Each image becomes one vector: the deepest backbone feature map, global-average-pooled.
The score is the Mahalanobis distance of that vector to the distribution of the
training vectors.

Limitation: the score is image-level only. There is no spatial evidence, so the
anomaly map is the image score repeated at every pixel and carries no localization.
Pixel-level metrics for this model therefore measure nothing beyond its image score.

With far fewer training images than feature dimensions the sample covariance is
singular, so it is estimated with Ledoit-Wolf shrinkage, which needs no tuned
parameter and sees training data only.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config
from src.models.backbone import build_backbone, resolve_out_indices

#: Stride of the deepest ResNet stage (`layer4`).
DEEPEST_STRIDE = 32


class MahalanobisModel:
    name = "mahalanobis"

    def __init__(
        self,
        backbone_name: str,
        out_indices: list[int],
        feature_map_size: tuple[int, int],
        input_size: int,
        *,
        module: nn.Module | None = None,
        pretrained: bool = True,
    ) -> None:
        self.backbone_name = backbone_name
        self.out_indices = list(out_indices)
        self.feature_map_size = feature_map_size
        self.input_size = input_size
        self.backbone = (
            module
            if module is not None
            else build_backbone(backbone_name, self.out_indices, pretrained=pretrained)
        )
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: Config) -> MahalanobisModel:
        input_size = config.preprocessing.center_crop
        out_indices, feature_map_size = resolve_out_indices(
            config.model.backbone, (DEEPEST_STRIDE,), input_size
        )
        return cls(config.model.backbone, out_indices, feature_map_size, input_size)

    def _embed(self, images: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            feature_map = self.backbone(images)[-1]
            pooled = feature_map.mean(dim=(2, 3))
        return pooled.double().numpy()

    def fit(self, loader: DataLoader) -> None:
        embeddings = np.concatenate([self._embed(batch["image"]) for batch in loader], axis=0)
        estimator = LedoitWolf().fit(embeddings)
        self.mean = estimator.location_.astype(np.float32)
        self.precision = estimator.precision_.astype(np.float32)

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.precision is None:
            raise RuntimeError("model is not fitted")
        centered = self._embed(images) - self.mean.astype(np.float64)
        squared = np.einsum("bi,ij,bj->b", centered, self.precision.astype(np.float64), centered)
        scores = np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)
        height = width = self.input_size
        maps = np.broadcast_to(scores[:, None, None], (len(scores), height, width)).copy()
        return scores, maps

    def get_state(self) -> dict[str, Any]:
        """Everything `predict` needs beyond the (re-creatable) pretrained backbone."""
        if self.mean is None or self.precision is None:
            raise RuntimeError("model is not fitted")
        return {
            "backbone": self.backbone_name,
            "out_indices": self.out_indices,
            "feature_map_size": self.feature_map_size,
            "input_size": self.input_size,
            "mean": self.mean,
            "precision": self.precision,
        }

    @classmethod
    def from_state(
        cls, state: dict[str, Any], *, module: nn.Module | None = None, pretrained: bool = True
    ) -> MahalanobisModel:
        model = cls(
            state["backbone"],
            state["out_indices"],
            tuple(state["feature_map_size"]),
            state["input_size"],
            module=module,
            pretrained=pretrained,
        )
        model.mean = state["mean"]
        model.precision = state["precision"]
        return model

    def save(self, path: str) -> None:
        joblib.dump(self.get_state(), path)

    @classmethod
    def load(cls, path: str) -> MahalanobisModel:
        return cls.from_state(joblib.load(path))
