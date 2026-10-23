"""Tests for the README generator (T12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.readme import BLOCKS, TBD, body_of, build, inject

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = "intro\n<!-- BEGIN a -->\nold\n<!-- END a -->\noutro\n"


def test_inject_replaces_only_the_block_content() -> None:
    result = inject(TEMPLATE, {"a": "new table"})
    assert result == "intro\n<!-- BEGIN a -->\nnew table\n<!-- END a -->\noutro\n"


def test_inject_is_idempotent() -> None:
    once = inject(TEMPLATE, {"a": "new table"})
    assert inject(once, {"a": "new table"}) == once


def test_inject_fails_loudly_on_a_missing_block() -> None:
    with pytest.raises(ValueError, match="no 'b' block"):
        inject(TEMPLATE, {"b": "x"})


def test_body_drops_the_title_line(tmp_path: Path) -> None:
    results = tmp_path / "metrics.md"
    results.write_text("# Title\n\ntable here\n", encoding="utf-8")
    assert body_of(results) == "table here"


def test_missing_results_file_reads_tbd(tmp_path: Path) -> None:
    assert body_of(tmp_path / "absent.md") == TBD


def test_committed_readme_matches_results() -> None:
    """The README tables must be exactly what the result files say."""
    readme_path = PROJECT_ROOT / "README.md"
    assert build(readme_path, PROJECT_ROOT) == readme_path.read_text(encoding="utf-8")
    for relative in BLOCKS.values():
        assert (PROJECT_ROOT / relative).is_file(), f"{relative} is missing"
