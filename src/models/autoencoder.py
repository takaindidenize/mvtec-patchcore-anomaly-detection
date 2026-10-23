"""Convolutional autoencoder baseline.

Trained to reconstruct defect-free images through a narrow bottleneck. A defect is
something the network never learned to draw, so it reconstructs badly there: the
anomaly map is the per-pixel squared reconstruction error, averaged over channels,
and the image score is the maximum of that map.

The training hyperparameters are named constants rather than config keys, because
the section 11 schema is declared complete. They are fixed up front and were not
tuned against any test data.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config

LOGGER = logging.getLogger(__name__)

EPOCHS = 50
LEARNING_RATE = 1e-3
LATENT_CHANNELS = 128
#: Channel widths of the four hidden encoder stages; a fifth stage maps to the latent.
HIDDEN_CHANNELS = (32, 64, 128, 128)


def _down(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def _up(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class _ConvAutoencoder(nn.Module):
    """Five stride-2 stages down (224 -> 7) and five back up."""

    def __init__(self) -> None:
        super().__init__()
        widths = (3, *HIDDEN_CHANNELS)
        encoder: list[nn.Module] = []
        for in_channels, out_channels in zip(widths[:-1], widths[1:], strict=True):
            encoder += _down(in_channels, out_channels)
        encoder += _down(widths[-1], LATENT_CHANNELS)

        reversed_widths = (LATENT_CHANNELS, *reversed(HIDDEN_CHANNELS))
        decoder: list[nn.Module] = []
        for in_channels, out_channels in zip(
            reversed_widths[:-1], reversed_widths[1:], strict=True
        ):
            decoder += _up(in_channels, out_channels)
        # The output stays linear: inputs are mean/std normalized, not bounded to [0, 1].
        decoder.append(
            nn.ConvTranspose2d(reversed_widths[-1], 3, kernel_size=4, stride=2, padding=1)
        )
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))


class AutoencoderModel:
    name = "autoencoder"
    #: The autoencoder is trained by SGD, so `train_fit` is shuffled (with a seeded generator).
    shuffle_train_fit = True
    #: No pretrained backbone is involved.
    backbone_name = "none"
    out_indices: list[int] = []

    def __init__(self, input_size: int, *, epochs: int = EPOCHS) -> None:
        self.input_size = input_size
        self.epochs = epochs
        self.network = _ConvAutoencoder()
        self.feature_map_size = (input_size // 32, input_size // 32)
        self.fitted = False

    @classmethod
    def from_config(cls, config: Config) -> AutoencoderModel:
        # Weight initialization draws from the global generator, so seed it explicitly.
        torch.manual_seed(config.seed)
        return cls(config.preprocessing.center_crop)

    def fit(self, loader: DataLoader) -> None:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.network.train()
        for epoch in range(self.epochs):
            total, count = 0.0, 0
            for batch in loader:
                images = batch["image"]
                loss = nn.functional.mse_loss(self.network(images), images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(images)
                count += len(images)
            LOGGER.info("epoch %d/%d: train mse %.5f", epoch + 1, self.epochs, total / count)
        self.network.eval()
        self.fitted = True

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        self.network.eval()
        with torch.inference_mode():
            error = (self.network(images) - images).pow(2).mean(dim=1)
        maps = error.numpy().astype(np.float32)
        scores = maps.reshape(len(maps), -1).max(axis=1)
        return scores, maps

    def get_state(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        return {
            "input_size": self.input_size,
            "state_dict": {key: value.clone() for key, value in self.network.state_dict().items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> AutoencoderModel:
        model = cls(state["input_size"])
        model.network.load_state_dict(state["state_dict"])
        model.network.eval()
        model.fitted = True
        return model

    def save(self, path: str) -> None:
        joblib.dump(self.get_state(), path)

    @classmethod
    def load(cls, path: str) -> AutoencoderModel:
        return cls.from_state(joblib.load(path))
