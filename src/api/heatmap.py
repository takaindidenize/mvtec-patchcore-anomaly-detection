"""Render an anomaly map as a colormap overlaid on the preprocessed input.

Pure numpy and Pillow: matplotlib is a development dependency and never enters the
serving path.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

OVERLAY_ALPHA = 0.5


def _colormap(values: np.ndarray) -> np.ndarray:
    """A jet-style colormap, (H, W) in [0, 1] -> (H, W, 3) uint8."""
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def render_heatmap_png_base64(
    image: torch.Tensor,
    anomaly_map: np.ndarray,
    threshold: float,
    mean: list[float],
    std: list[float],
) -> str:
    """Base64 PNG of the anomaly map blended over the preprocessed input.

    Args:
        image: The preprocessed input, (3, H, W), still normalized.
        anomaly_map: (H, W), on the same scale as the image score.
        threshold: The calibrated threshold. It sits at the middle of the colour
            scale, so the same colour means the same thing for every image, and
            anything red is above the alarm line.
        mean, std: The normalization the image went through, read from the bundle.
    """
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    base = (image * std_tensor + mean_tensor).clamp(0.0, 1.0)
    base_rgb = (base.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    scale = 2.0 * threshold if threshold > 0 else 1.0
    colored = _colormap(np.clip(anomaly_map / scale, 0.0, 1.0))
    blended = (1.0 - OVERLAY_ALPHA) * base_rgb + OVERLAY_ALPHA * colored

    buffer = io.BytesIO()
    Image.fromarray(blended.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
