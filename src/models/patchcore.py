"""PatchCore (AGENTS.md section 8.1).

Normal images are embedded into locally aggregated mid-level patch features. A greedy
k-center coreset of those patches is the memory bank. A test patch is scored by its
distance to the nearest memory-bank patch; the patch scores form a low-resolution map
that is upsampled and smoothed, and the image score is the maximum of that map.

The backbone is inference-only: no gradients and no fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config
from src.models.backbone import build_backbone, resolve_out_indices

LOGGER = logging.getLogger(__name__)

#: ResNet `layer2` and `layer3`.
FEATURE_STRIDES = (8, 16)

BYTES_PER_MB = 1024 * 1024


def greedy_kcenter(features: torch.Tensor, size: int, seed: int) -> np.ndarray:
    """Indices of a greedy k-center coreset of `features` (N, D).

    Starts from a seeded random point, then repeatedly adds the point farthest from
    the points chosen so far, so the coreset covers the full set.
    """
    count = features.shape[0]
    size = min(size, count)
    first = int(np.random.default_rng(seed).integers(count))

    squared_norms = features.pow(2).sum(dim=1)
    selected = [first]
    nearest = _squared_distances(features, squared_norms, first)
    for step in range(1, size):
        index = int(torch.argmax(nearest))
        selected.append(index)
        nearest = torch.minimum(nearest, _squared_distances(features, squared_norms, index))
        if step % 250 == 0:
            LOGGER.info("coreset %d/%d", step, size)
    return np.asarray(selected, dtype=np.int64)


def _squared_distances(features: torch.Tensor, squared_norms: torch.Tensor, index: int):
    distances = squared_norms - 2.0 * (features @ features[index]) + squared_norms[index]
    return distances.clamp_(min=0.0)


class PatchCoreModel:
    name = "patchcore"
    shuffle_train_fit = False

    def __init__(
        self,
        backbone_name: str,
        out_indices: list[int],
        input_size: int,
        coreset_ratio: float,
        gaussian_sigma: int,
        seed: int,
        *,
        module: nn.Module | None = None,
        pretrained: bool = True,
    ) -> None:
        self.backbone_name = backbone_name
        self.out_indices = list(out_indices)
        self.input_size = input_size
        self.coreset_ratio = coreset_ratio
        self.gaussian_sigma = gaussian_sigma
        self.seed = seed
        self.backbone = (
            module
            if module is not None
            else build_backbone(backbone_name, self.out_indices, pretrained=pretrained)
        )
        self.memory_bank: np.ndarray | None = None
        self.feature_map_size: tuple[int, int] = (0, 0)
        self._index: NearestNeighbors | None = None

    @classmethod
    def from_config(cls, config: Config) -> PatchCoreModel:
        input_size = config.preprocessing.center_crop
        out_indices, _ = resolve_out_indices(config.model.backbone, FEATURE_STRIDES, input_size)
        return cls(
            config.model.backbone,
            out_indices,
            input_size,
            config.model.coreset_ratio,
            config.model.gaussian_sigma,
            config.seed,
        )

    @property
    def memory_bank_mb(self) -> float:
        if self.memory_bank is None:
            raise RuntimeError("model is not fitted")
        return self.memory_bank.nbytes / BYTES_PER_MB

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """Locally aggregated features, (B, C, H, W) at the shallower layer's resolution."""
        with torch.inference_mode():
            shallow, deep = self.backbone(images)
            deep = F.interpolate(
                deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False
            )
            fused = torch.cat([shallow, deep], dim=1)
            return F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _patches(features: torch.Tensor) -> torch.Tensor:
        channels = features.shape[1]
        return features.permute(0, 2, 3, 1).reshape(-1, channels)

    def fit(self, loader: DataLoader) -> None:
        patches = []
        for batch in loader:
            # Each batch is embedded exactly once.
            features = self._embed(batch["image"])
            self.feature_map_size = (int(features.shape[-2]), int(features.shape[-1]))
            patches.append(self._patches(features).clone())
        all_patches = torch.cat(patches, dim=0)
        del patches

        size = max(1, int(self.coreset_ratio * all_patches.shape[0]))
        LOGGER.info("selecting %d of %d patches", size, all_patches.shape[0])
        selected = greedy_kcenter(all_patches, size, self.seed)
        self.memory_bank = all_patches[selected].numpy().astype(np.float32)
        self._build_index()

    def _build_index(self) -> None:
        assert self.memory_bank is not None
        self._index = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(self.memory_bank)

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.memory_bank is None or self._index is None:
            raise RuntimeError("model is not fitted")

        features = self._embed(images)
        batch = features.shape[0]
        height, width = features.shape[-2:]
        distances, _ = self._index.kneighbors(self._patches(features).numpy())
        patch_scores = torch.from_numpy(distances[:, 0].astype(np.float32)).reshape(
            batch, 1, height, width
        )

        upsampled = F.interpolate(
            patch_scores,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        maps = np.stack(
            [gaussian_filter(single, sigma=self.gaussian_sigma) for single in upsampled]
        ).astype(np.float32)
        scores = maps.reshape(batch, -1).max(axis=1)
        return scores, maps

    def get_state(self) -> None:
        """PatchCore keeps everything in the bundle's `memory_bank`, so there is no extra state."""
        return None

    @classmethod
    def from_bundle(
        cls, bundle: dict[str, Any], *, module: nn.Module | None = None, pretrained: bool = True
    ) -> PatchCoreModel:
        model = cls(
            bundle["backbone"],
            bundle["resolved_out_indices"],
            bundle["preprocessing"]["center_crop"],
            bundle["coreset_ratio"],
            bundle["gaussian_sigma"],
            bundle["seed"],
            module=module,
            pretrained=pretrained,
        )
        model.memory_bank = bundle["memory_bank"]
        model.feature_map_size = tuple(bundle["feature_map_size"])
        model._build_index()
        return model

    def save(self, path: str) -> None:
        if self.memory_bank is None:
            raise RuntimeError("model is not fitted")
        joblib.dump(
            {
                "backbone": self.backbone_name,
                "resolved_out_indices": self.out_indices,
                "preprocessing": {"center_crop": self.input_size},
                "coreset_ratio": self.coreset_ratio,
                "gaussian_sigma": self.gaussian_sigma,
                "seed": self.seed,
                "memory_bank": self.memory_bank,
                "feature_map_size": self.feature_map_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> PatchCoreModel:
        return cls.from_bundle(joblib.load(path))
