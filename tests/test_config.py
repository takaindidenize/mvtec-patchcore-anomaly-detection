"""Tests for the config loader (AGENTS.md section 11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "bottle.yaml"


def _raw_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_bottle_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config.category == "bottle"
    assert config.data_root == "data/mvtec"
    assert config.model.name in {"mahalanobis", "autoencoder", "patchcore"}


def test_preprocessing_matches_the_values_fixed_in_section_7() -> None:
    """Section 7 fixes these. This test fails if anyone drifts them without asking."""
    preprocessing = load_config(CONFIG_PATH).preprocessing
    assert preprocessing.resize == 256
    assert preprocessing.center_crop == 224
    assert preprocessing.mean == (0.485, 0.456, 0.406)
    assert preprocessing.std == (0.229, 0.224, 0.225)


def test_seed_override_changes_nothing_else() -> None:
    base = load_config(CONFIG_PATH)
    overridden = load_config(CONFIG_PATH, seed=2)
    assert overridden.seed == 2
    assert base.seed != overridden.seed
    assert overridden.preprocessing == base.preprocessing
    assert overridden.model == base.model
    assert overridden.training == base.training


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["learning_rate"] = 0.001
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write(tmp_path, raw))


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["model"]["epochs"] = 10
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write(tmp_path, raw))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    del raw["target_fpr"]
    with pytest.raises(ValueError, match="missing keys"):
        load_config(_write(tmp_path, raw))


def test_missing_section_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config()
    del raw["preprocessing"]
    with pytest.raises(ValueError, match="preprocessing"):
        load_config(_write(tmp_path, raw))
