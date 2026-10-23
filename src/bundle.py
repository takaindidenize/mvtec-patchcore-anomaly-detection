"""The single artifact file that serving loads (AGENTS.md section 12).

`model_state` is the one key beyond the section 12 list: Mahalanobis statistics and
autoencoder weights have nowhere else to live, and serving may read only this file
(approved by the user, see `docs/DECISIONS.md`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import joblib

SCHEMA_VERSION = 1

BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "category",
        "model_name",
        "backbone",
        "resolved_out_indices",
        "preprocessing",
        "feature_map_size",
        "memory_bank",
        "coreset_ratio",
        "gaussian_sigma",
        "threshold",
        "target_fpr",
        "seed",
        "created_at",
        "git_commit",
        "model_state",
    }
)


def bundle_path(artifacts_dir: str | Path, category: str, model_name: str, seed: int) -> Path:
    return Path(artifacts_dir) / f"{category}_{model_name}_seed{seed}.joblib"


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def write_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    if set(bundle) != BUNDLE_KEYS:
        missing, extra = BUNDLE_KEYS - set(bundle), set(bundle) - BUNDLE_KEYS
        raise ValueError(f"bundle keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if set(bundle) != BUNDLE_KEYS:
        raise ValueError(f"{path}: not a valid bundle, keys are {sorted(bundle)}")
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {bundle['schema_version']}")
    return bundle
