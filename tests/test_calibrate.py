import numpy as np
import pytest

from src.calibrate import calibrate_threshold


@pytest.mark.parametrize("target_fpr", [0.01, 0.05, 0.1])
def test_threshold_is_the_target_quantile(target_fpr: float) -> None:
    scores = np.random.default_rng(1).normal(size=500)
    expected = np.quantile(scores, 1.0 - target_fpr)
    assert calibrate_threshold(scores, target_fpr) == expected


def test_threshold_leaves_roughly_the_target_fraction_above_it() -> None:
    scores = np.random.default_rng(2).normal(size=20000)
    threshold = calibrate_threshold(scores, 0.01)
    assert (scores > threshold).mean() == pytest.approx(0.01, abs=0.003)


def test_a_lower_target_fpr_gives_a_higher_threshold() -> None:
    scores = np.random.default_rng(3).normal(size=500)
    assert calibrate_threshold(scores, 0.01) > calibrate_threshold(scores, 0.1)


@pytest.mark.parametrize("target_fpr", [0.0, 1.0, -0.1, 1.5])
def test_invalid_target_fpr_is_rejected(target_fpr: float) -> None:
    with pytest.raises(ValueError, match="target_fpr"):
        calibrate_threshold(np.zeros(10), target_fpr)


def test_empty_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one validation score"):
        calibrate_threshold(np.array([]), 0.01)
