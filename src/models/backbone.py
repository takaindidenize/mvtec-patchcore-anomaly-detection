"""Pretrained feature extractor shared by the models.

Layer indices are never taken from memory (AGENTS.md section 8.1). They are resolved
by running one dummy forward pass, printing the output shapes and matching each
feature map's stride against the strides the caller asks for.
"""

from __future__ import annotations

import logging

import timm
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


def resolve_out_indices(
    backbone: str, strides: tuple[int, ...], input_size: int
) -> tuple[list[int], tuple[int, int]]:
    """Find the `features_only` indices whose maps have the requested strides.

    Returns:
        The resolved indices, in the order of `strides`, and the spatial size of the
        feature map at the last requested stride.
    """
    # Weights do not change a layer's shape, so the probe skips the download.
    probe = timm.create_model(backbone, pretrained=False, features_only=True).eval()
    with torch.inference_mode():
        outputs = probe(torch.zeros(1, 3, input_size, input_size))

    stride_to_index: dict[int, int] = {}
    for index, feature in enumerate(outputs):
        stride = input_size // feature.shape[-1]
        stride_to_index[stride] = index
        LOGGER.info(
            "%s feature %d: shape=%s stride=%d", backbone, index, tuple(feature.shape), stride
        )
        print(f"{backbone} feature {index}: shape={tuple(feature.shape)} stride={stride}")

    missing = [stride for stride in strides if stride not in stride_to_index]
    if missing:
        raise ValueError(f"{backbone} has no feature map at stride(s) {missing}")

    indices = [stride_to_index[stride] for stride in strides]
    last = outputs[indices[-1]]
    LOGGER.info("resolved out_indices %s for strides %s", indices, list(strides))
    print(f"resolved out_indices {indices} for strides {list(strides)}")
    return indices, (int(last.shape[-2]), int(last.shape[-1]))


def build_backbone(backbone: str, out_indices: list[int], *, pretrained: bool = True) -> nn.Module:
    """A frozen, inference-only feature extractor returning the selected maps."""
    model = timm.create_model(
        backbone, pretrained=pretrained, features_only=True, out_indices=tuple(out_indices)
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
