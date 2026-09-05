"""MVTec AD dataset, deterministic splits and dataloaders.

Implements the data contract of AGENTS.md section 5, the splits of section 6 and
the preprocessing of section 7. Preprocessing lives here and only here; serving
code reads its parameters from the artifact bundle rather than restating them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from src.config import Config, PreprocessingConfig

#: Label convention of section 5.
NORMAL_LABEL = 0
ANOMALOUS_LABEL = 1

#: Name of the defect-free subdirectory in both `train/` and `test/`.
GOOD_DIR = "good"

#: MVTec AD ships every image and every mask as PNG.
IMAGE_SUFFIX = "*.png"


@dataclass(frozen=True)
class Sample:
    """One image, its mask if it has one, and its label."""

    image_path: Path
    #: `None` for defect-free images, which section 5 defines as an all-zero mask.
    mask_path: Path | None
    label: int
    defect_type: str


@dataclass(frozen=True)
class Splits:
    """The three splits of section 6."""

    train_fit: list[Sample]
    val_normal: list[Sample]
    test: list[Sample]


@dataclass(frozen=True)
class DataLoaders:
    train_fit: DataLoader
    val_normal: DataLoader
    test: DataLoader


def _normal_sample(path: Path) -> Sample:
    return Sample(image_path=path, mask_path=None, label=NORMAL_LABEL, defect_type=GOOD_DIR)


def list_train_good(category_root: Path) -> list[Path]:
    """Every defect-free training image, in a stable order."""
    train_good = category_root / "train" / GOOD_DIR
    if not train_good.is_dir():
        raise FileNotFoundError(f"expected defect-free training images at {train_good}")
    return sorted(train_good.glob(IMAGE_SUFFIX))


def list_test_samples(category_root: Path) -> list[Sample]:
    """Every test image with its mask, in a stable order.

    `test/good/` images carry no mask file and are represented with `mask_path=None`;
    they are neither skipped nor treated as an error.
    """
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    if not test_root.is_dir():
        raise FileNotFoundError(f"expected a test split at {test_root}")

    samples: list[Sample] = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        is_good = defect_type == GOOD_DIR
        for image_path in sorted(defect_dir.glob(IMAGE_SUFFIX)):
            if is_good:
                samples.append(_normal_sample(image_path))
                continue
            # Section 5 pairing rule: `000.png` <-> `000_mask.png`.
            mask_path = ground_truth_root / defect_type / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"{image_path} has no mask at {mask_path}")
            samples.append(
                Sample(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=ANOMALOUS_LABEL,
                    defect_type=defect_type,
                )
            )
    return samples


def split_train_good(
    paths: Sequence[Path], validation_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split `train/good` into `train_fit` and `val_normal`.

    Deterministic given `seed`, and the two results are disjoint by construction.
    `val_normal` exists only to calibrate the threshold (section 9); it contains no
    anomalies and is never used to fit a model.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")

    ordered = sorted(paths)
    validation_size = round(len(ordered) * validation_ratio)
    if validation_size == 0 or validation_size == len(ordered):
        raise ValueError(
            f"validation_ratio {validation_ratio} leaves an empty split "
            f"for {len(ordered)} training images"
        )

    # An explicit local generator; section 2 rule 10 forbids implicit global randomness.
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    val_normal = sorted(ordered[index] for index in permutation[:validation_size])
    train_fit = sorted(ordered[index] for index in permutation[validation_size:])
    return train_fit, val_normal


def preprocess_image(image: Image.Image, preprocessing: PreprocessingConfig) -> torch.Tensor:
    """Section 7: RGB, resize shorter side, center crop, scale to [0, 1], normalize."""
    image = image.convert("RGB")
    image = TF.resize(image, preprocessing.resize, interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [preprocessing.center_crop, preprocessing.center_crop])
    tensor = TF.pil_to_tensor(image).float().div_(255.0)
    return TF.normalize(tensor, list(preprocessing.mean), list(preprocessing.std))


def preprocess_mask(
    mask: Image.Image | None,
    image_size: tuple[int, int],
    preprocessing: PreprocessingConfig,
) -> torch.Tensor:
    """Section 7: masks get resize and center crop only, never normalization.

    Interpolation is nearest-neighbour throughout; bilinear would produce non-binary
    values and silently corrupt the pixel-level ground truth.

    Args:
        mask: The mask image, or `None` for a defect-free image.
        image_size: The source image's `(width, height)`, used to detect a size mismatch.
    """
    crop = preprocessing.center_crop
    if mask is None:
        # Section 5 defines this as an all-zero array of the image's spatial size.
        # Resizing and cropping all zeros yields all zeros, so build it at the final size.
        return torch.zeros((crop, crop), dtype=torch.float32)

    mask = mask.convert("L")
    if mask.size != image_size:
        # Section 5: align the mask to its image before the shared pipeline.
        width, height = image_size
        mask = TF.resize(mask, [height, width], interpolation=InterpolationMode.NEAREST)

    mask = TF.resize(mask, preprocessing.resize, interpolation=InterpolationMode.NEAREST)
    mask = TF.center_crop(mask, [crop, crop])
    return torch.from_numpy((np.array(mask) > 0).astype(np.float32))


class MVTecDataset(Dataset):
    """Yields preprocessed images with their masks and labels."""

    def __init__(self, samples: Sequence[Sample], preprocessing: PreprocessingConfig) -> None:
        self.samples = list(samples)
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image.load()
            image_size = image.size
            image_tensor = preprocess_image(image, self.preprocessing)

        if sample.mask_path is None:
            mask_tensor = preprocess_mask(None, image_size, self.preprocessing)
        else:
            with Image.open(sample.mask_path) as mask:
                mask.load()
                mask_tensor = preprocess_mask(mask, image_size, self.preprocessing)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": str(sample.image_path),
            "defect_type": sample.defect_type,
        }


def build_splits(config: Config) -> Splits:
    """Build the three splits of section 6 for the configured category."""
    category_root = config.category_root
    train_fit_paths, val_normal_paths = split_train_good(
        list_train_good(category_root), config.validation_ratio, config.seed
    )
    return Splits(
        train_fit=[_normal_sample(path) for path in train_fit_paths],
        val_normal=[_normal_sample(path) for path in val_normal_paths],
        test=list_test_samples(category_root),
    )


def build_dataloaders(config: Config, *, shuffle_train_fit: bool = False) -> DataLoaders:
    """Build one dataloader per split.

    Only `train_fit` is ever shuffled, and only when a model needs it (the
    autoencoder does, PatchCore does not). Shuffling draws from an explicitly
    seeded generator, never from global randomness.
    """
    splits = build_splits(config)

    def make_loader(samples: Sequence[Sample], shuffle: bool) -> DataLoader:
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(config.seed)
        return DataLoader(
            MVTecDataset(samples, config.preprocessing),
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            num_workers=config.training.num_workers,
            generator=generator,
        )

    return DataLoaders(
        train_fit=make_loader(splits.train_fit, shuffle_train_fit),
        val_normal=make_loader(splits.val_normal, False),
        test=make_loader(splits.test, False),
    )
