"""Shared test fixtures."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """`data_root` in the config is project-relative, so pin the working directory."""
    monkeypatch.chdir(PROJECT_ROOT)
