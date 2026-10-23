"""Fill the result tables of `README.md` from `results/`.

AGENTS.md rule 3: no number in a markdown file is typed by hand. The README prose
therefore carries no measured values; each results section is a marked block that
this script replaces with the body of a file the other scripts wrote.

    python -m src.readme            rewrite the blocks in README.md
    python -m src.readme --check    exit non-zero if README.md is out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Marker name -> the results file whose body is embedded there.
BLOCKS = {
    "metrics": "results/metrics.md",
    "coreset": "results/coreset.md",
    "latency": "results/latency.md",
}

TBD = "TBD: this file has not been generated yet."


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)",
        flags=re.DOTALL,
    )


def body_of(results_file: Path) -> str:
    """The generated file without its title line, which the README supplies itself."""
    if not results_file.is_file():
        return TBD
    lines = results_file.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def inject(readme: str, blocks: dict[str, str]) -> str:
    """Replace the content of every `<!-- BEGIN name --> ... <!-- END name -->` block."""
    for name, content in blocks.items():
        pattern = _pattern(name)
        if not pattern.search(readme):
            raise ValueError(f"README.md has no '{name}' block")
        readme = pattern.sub(
            lambda match, text=content: f"{match.group(1)}{text}\n{match.group(2)}", readme
        )
    return readme


def build(readme_path: Path, root: Path) -> str:
    blocks = {name: body_of(root / relative) for name, relative in BLOCKS.items()}
    return inject(readme_path.read_text(encoding="utf-8"), blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    readme_path = root / "README.md"
    current = readme_path.read_text(encoding="utf-8")
    updated = build(readme_path, root)

    if args.check:
        if current != updated:
            print("README.md is out of date; run `python -m src.readme`", file=sys.stderr)
            raise SystemExit(1)
        print("README.md is up to date")
        return

    readme_path.write_text(updated, encoding="utf-8", newline="\n")
    print("updated README.md")


if __name__ == "__main__":
    main()
