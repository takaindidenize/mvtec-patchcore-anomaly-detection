"""Tests for the dataset, splits and dataloaders (AGENTS.md sections 5-7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.data.mvtec import (
    ANOMALOUS_LABEL,
    NORMAL_LABEL,
    MVTecDataset,
    build_dataloaders,
    build_splits,
    list_train_good,
    split_train_good,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "bottle.yaml")
CATEGORY_ROOT = PROJECT_ROOT / CONFIG.category_root

requires_dataset = pytest.mark.skipif(
    not (CATEGORY_ROOT / "train" / "good").is_dir(),
    reason=f"MVTec AD not found at {CATEGORY_ROOT}; the dataset is never committed",
)


def _fake_paths(count: int) -> list[Path]:
    return [Path(f"train/good/{index:03d}.png") for index in range(count)]


# --- Splits (section 6): pure logic, no dataset required -------------------------


def test_split_is_disjoint_and_loses_nothing() -> None:
    paths = _fake_paths(209)
    train_fit, val_normal = split_train_good(paths, 0.2, seed=0)
    assert set(train_fit).isdisjoint(val_normal)
    assert sorted(train_fit + val_normal) == sorted(paths)


def test_split_sizes_follow_the_ratio() -> None:
    paths = _fake_paths(209)
    train_fit, val_normal = split_train_good(paths, 0.2, seed=0)
    assert len(val_normal) == round(len(paths) * 0.2)
    assert len(train_fit) == len(paths) - len(val_normal)


def test_split_is_deterministic_given_the_seed() -> None:
    paths = _fake_paths(209)
    assert split_train_good(paths, 0.2, seed=0) == split_train_good(paths, 0.2, seed=0)


def test_split_ignores_the_order_of_its_input() -> None:
    paths = _fake_paths(209)
    assert split_train_good(paths, 0.2, seed=0) == split_train_good(paths[::-1], 0.2, seed=0)


def test_different_seeds_give_different_splits() -> None:
    paths = _fake_paths(209)
    assert split_train_good(paths, 0.2, seed=0) != split_train_good(paths, 0.2, seed=1)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_invalid_validation_ratio_is_rejected(ratio: float) -> None:
    with pytest.raises(ValueError, match="validation_ratio"):
        split_train_good(_fake_paths(10), ratio, seed=0)


def test_ratio_that_would_empty_a_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty split"):
        split_train_good(_fake_paths(4), 0.01, seed=0)


# --- Dataset (sections 5 and 7): requires the extracted dataset ------------------


@requires_dataset
def test_splits_partition_train_good() -> None:
    splits = build_splits(CONFIG)
    train_good = set(list_train_good(CONFIG.category_root))
    fit_paths = {sample.image_path for sample in splits.train_fit}
    val_paths = {sample.image_path for sample in splits.val_normal}

    assert fit_paths.isdisjoint(val_paths)
    assert fit_paths | val_paths == train_good
    assert len(val_paths) == round(len(train_good) * CONFIG.validation_ratio)


@requires_dataset
def test_train_and_validation_hold_no_anomalies() -> None:
    splits = build_splits(CONFIG)
    for sample in splits.train_fit + splits.val_normal:
        assert sample.label == NORMAL_LABEL
        assert sample.mask_path is None


@requires_dataset
def test_test_split_counts_match_the_directories() -> None:
    splits = build_splits(CONFIG)
    on_disk = sorted((CONFIG.category_root / "test").rglob("*.png"))
    assert len(splits.test) == len(on_disk)
    assert {sample.image_path for sample in splits.test} == set(on_disk)
    assert {sample.label for sample in splits.test} == {NORMAL_LABEL, ANOMALOUS_LABEL}


@requires_dataset
def test_image_and_mask_shapes_and_dtypes() -> None:
    splits = build_splits(CONFIG)
    dataset = MVTecDataset(splits.test, CONFIG.preprocessing)
    crop = CONFIG.preprocessing.center_crop

    for index in (0, len(dataset) - 1):
        item = dataset[index]
        assert item["image"].shape == (3, crop, crop)
        assert item["image"].dtype == torch.float32
        assert item["mask"].shape == (crop, crop)
        assert item["mask"].dtype == torch.float32


@requires_dataset
def test_masks_are_binary() -> None:
    splits = build_splits(CONFIG)
    dataset = MVTecDataset(splits.test, CONFIG.preprocessing)

    for index in range(0, len(dataset), max(1, len(dataset) // 10)):
        assert set(torch.unique(dataset[index]["mask"]).tolist()) <= {0.0, 1.0}


@requires_dataset
def test_defect_free_test_images_carry_an_all_zero_mask() -> None:
    splits = build_splits(CONFIG)
    good = [sample for sample in splits.test if sample.label == NORMAL_LABEL]
    dataset = MVTecDataset(good, CONFIG.preprocessing)

    assert len(good) > 0
    assert dataset[0]["mask"].sum() == 0


@requires_dataset
def test_anomalous_images_carry_a_non_empty_mask() -> None:
    splits = build_splits(CONFIG)
    anomalous = [sample for sample in splits.test if sample.label == ANOMALOUS_LABEL]
    dataset = MVTecDataset(anomalous, CONFIG.preprocessing)

    # A single defect can in principle fall outside the center crop, so this asserts
    # that masks survive preprocessing at all, not that every one of them is non-empty.
    assert any(dataset[index]["mask"].sum() > 0 for index in range(min(10, len(dataset))))


@requires_dataset
def test_dataloader_batches_have_the_right_shapes() -> None:
    loaders = build_dataloaders(CONFIG)
    batch = next(iter(loaders.test))
    batch_size = CONFIG.training.batch_size
    crop = CONFIG.preprocessing.center_crop

    assert batch["image"].shape == (batch_size, 3, crop, crop)
    assert batch["mask"].shape == (batch_size, crop, crop)
    assert batch["label"].shape == (batch_size,)


@requires_dataset
def test_dataloaders_cover_every_sample() -> None:
    splits = build_splits(CONFIG)
    loaders = build_dataloaders(CONFIG)

    for split, loader in (
        (splits.train_fit, loaders.train_fit),
        (splits.val_normal, loaders.val_normal),
        (splits.test, loaders.test),
    ):
        assert len(loader.dataset) == len(split)
