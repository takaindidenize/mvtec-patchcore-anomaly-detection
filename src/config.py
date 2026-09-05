"""Typed loader for `configs/<category>.yaml`.

AGENTS.md section 11 declares its schema complete: "this is the complete set of
keys; add none silently". Loading therefore rejects unknown and missing keys, so
a typo or a quietly added key fails loudly instead of changing behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessingConfig:
    resize: int
    center_crop: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backbone: str
    coreset_ratio: float
    gaussian_sigma: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: str
    results_dir: str


@dataclass(frozen=True)
class Config:
    category: str
    data_root: str
    seed: int
    validation_ratio: float
    target_fpr: float
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    @property
    def category_root(self) -> Path:
        return Path(self.data_root) / self.category


_SECTIONS: dict[str, type] = {
    "preprocessing": PreprocessingConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "output": OutputConfig,
}


def _build(cls: type, raw: dict[str, Any], where: str) -> Any:
    """Instantiate `cls` from `raw`, rejecting any key mismatch."""
    expected = {field.name for field in fields(cls)}
    provided = set(raw)

    problems = []
    if missing := sorted(expected - provided):
        problems.append(f"missing keys {missing}")
    if unknown := sorted(provided - expected):
        problems.append(f"unknown keys {unknown}")
    if problems:
        raise ValueError(f"{where}: " + ", ".join(problems))

    return cls(**raw)


def load_config(path: str | Path, *, seed: int | None = None) -> Config:
    """Load and validate a category config.

    Args:
        path: Path to `configs/<category>.yaml`.
        seed: Overrides the config's `seed`. Section 10 requires every model to be
            run with seeds 0, 1 and 2, while the section 11 schema holds a single
            `seed` key, so the sweep is driven from the command line rather than by
            adding a key the schema does not allow.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a top-level YAML mapping")

    parsed: dict[str, Any] = dict(raw)
    for name, section_cls in _SECTIONS.items():
        section = parsed.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"{path}: section '{name}' is missing or is not a mapping")
        section = dict(section)
        if name == "preprocessing":
            # YAML gives lists; the config is frozen and hashable, so use tuples.
            for key in ("mean", "std"):
                if isinstance(section.get(key), list):
                    section[key] = tuple(section[key])
        parsed[name] = _build(section_cls, section, f"{path} -> {name}")

    config: Config = _build(Config, parsed, str(path))
    if seed is not None:
        config = replace(config, seed=seed)
    return config
