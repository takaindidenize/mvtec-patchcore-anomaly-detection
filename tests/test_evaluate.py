import numpy as np
import pytest

from src.evaluate import (
    au_pro,
    evaluate,
    f1_max,
    fpr_at,
    image_auroc,
    pixel_auroc,
    recall_at,
)


def test_random_scores_give_chance_level_auroc() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 1000)
    assert image_auroc(labels, rng.random(labels.size)) == pytest.approx(0.5, abs=0.05)


def test_perfect_scores_give_auroc_of_one() -> None:
    labels = np.repeat([0, 1], 50)
    assert image_auroc(labels, labels.astype(float)) == 1.0


def test_inverted_scores_give_auroc_of_zero() -> None:
    labels = np.repeat([0, 1], 50)
    assert image_auroc(labels, 1.0 - labels) == 0.0


def test_auroc_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both normal and anomalous"):
        image_auroc(np.zeros(10), np.arange(10))


def test_pixel_auroc_is_one_when_the_map_matches_the_mask() -> None:
    masks = np.zeros((2, 16, 16))
    masks[0, 2:6, 2:6] = 1.0
    assert pixel_auroc(masks, masks.copy()) == 1.0


def test_au_pro_is_one_for_a_perfect_map() -> None:
    masks = np.zeros((2, 32, 32))
    masks[0, 4:12, 4:12] = 1.0
    masks[1, 20:24, 20:24] = 1.0
    assert au_pro(masks, masks.copy()) == pytest.approx(1.0)


def test_au_pro_catches_a_missed_small_region_that_pixel_auroc_forgives() -> None:
    """The reason AU-PRO exists: regions count equally, pixels do not."""
    masks = np.zeros((2, 64, 64))
    masks[0, 10:40, 10:40] = 1.0  # large defect, found
    masks[1, 5:8, 5:8] = 1.0  # small defect, missed entirely

    maps = np.zeros_like(masks)
    maps[0, 10:40, 10:40] = 1.0

    assert pixel_auroc(masks, maps) > 0.95
    assert au_pro(masks, maps) < pixel_auroc(masks, maps) - 0.25


def test_au_pro_needs_a_defect() -> None:
    with pytest.raises(ValueError, match="defect region"):
        au_pro(np.zeros((1, 8, 8)), np.zeros((1, 8, 8)))


def test_au_pro_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        au_pro(np.ones((1, 8, 8)), np.ones((1, 4, 4)))


def test_f1_max_is_one_for_separable_scores() -> None:
    labels = np.repeat([0, 1], 20)
    assert f1_max(labels, labels.astype(float)) == 1.0


def test_recall_and_fpr_read_off_the_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.4, 0.8])

    assert recall_at(labels, scores, 0.5) == 0.5
    assert fpr_at(labels, scores, 0.5) == 0.5


def test_evaluate_returns_every_metric() -> None:
    labels = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    masks = np.zeros((2, 16, 16))
    masks[1, 4:8, 4:8] = 1.0
    maps = masks.copy()

    metrics = evaluate(labels, scores, masks, maps, threshold=0.5)

    assert set(metrics.as_dict()) == {
        "image_auroc",
        "pixel_auroc",
        "au_pro",
        "f1_max",
        "recall_at_threshold",
        "fpr_at_threshold",
    }
    assert metrics.image_auroc == 1.0
    assert metrics.recall_at_threshold == 1.0
    assert metrics.fpr_at_threshold == 0.0
