"""Audit of the extracted dataset: counts, the image/mask pairing rule, sizes, binarity.

Run once after extracting MVTec AD. Not used by training or serving.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.mvtec import GOOD_DIR, IMAGE_SUFFIX

MASK_SUFFIX = "_mask.png"
EXPECTED_MASK_VALUES = {0, 255}


@dataclass
class CategoryReport:
    category: str
    train_good: int = 0
    test_counts: dict[str, int] = field(default_factory=dict)
    mask_counts: dict[str, int] = field(default_factory=dict)
    image_sizes: Counter = field(default_factory=Counter)
    mask_value_sets: set[tuple[int, ...]] = field(default_factory=set)
    missing_masks: list[str] = field(default_factory=list)
    orphan_masks: list[str] = field(default_factory=list)
    size_mismatches: list[str] = field(default_factory=list)
    non_binary_masks: list[str] = field(default_factory=list)
    structure_problems: list[str] = field(default_factory=list)

    @property
    def findings(self) -> list[str]:
        problems: list[str] = []
        for label, entries in (
            ("missing mask", self.missing_masks),
            ("orphan mask", self.orphan_masks),
            ("image/mask size mismatch", self.size_mismatches),
            ("non-binary mask", self.non_binary_masks),
            ("unexpected structure", self.structure_problems),
        ):
            problems.extend(f"{self.category}: {label}: {entry}" for entry in entries)
        return problems


def _image_size(path: Path) -> tuple[int, int]:
    # Image.open only reads the header, so this stays cheap over thousands of files.
    with Image.open(path) as image:
        return image.size


def _mask_values(path: Path) -> set[int]:
    with Image.open(path) as mask:
        return set(np.unique(np.array(mask.convert("L"))).tolist())


def build_category_report(category_root: Path) -> CategoryReport:
    report = CategoryReport(category=category_root.name)

    train_good = category_root / "train" / GOOD_DIR
    if train_good.is_dir():
        report.train_good = len(list(train_good.glob(IMAGE_SUFFIX)))
    else:
        report.structure_problems.append(f"missing train/{GOOD_DIR}")

    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    if not test_root.is_dir():
        report.structure_problems.append("missing test/")
        return report

    test_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
    for defect_dir in test_dirs:
        defect = defect_dir.name
        images = sorted(defect_dir.glob(IMAGE_SUFFIX))
        report.test_counts[defect] = len(images)

        mask_dir = ground_truth_root / defect
        masks = sorted(mask_dir.glob(IMAGE_SUFFIX)) if mask_dir.is_dir() else []
        report.mask_counts[defect] = len(masks)

        for image_path in images:
            report.image_sizes[_image_size(image_path)] += 1

        if defect == GOOD_DIR:
            if masks:
                report.structure_problems.append(
                    f"ground_truth/{GOOD_DIR} holds {len(masks)} masks, expected none"
                )
            continue

        _audit_masks(report, defect, images, mask_dir, masks)

    if ground_truth_root.is_dir():
        known = {path.name for path in test_dirs}
        for mask_dir in sorted(path for path in ground_truth_root.iterdir() if path.is_dir()):
            if mask_dir.name not in known:
                report.structure_problems.append(
                    f"ground_truth/{mask_dir.name} has no test/ counterpart"
                )

    return report


def _audit_masks(
    report: CategoryReport,
    defect: str,
    images: list[Path],
    mask_dir: Path,
    masks: list[Path],
) -> None:
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}{MASK_SUFFIX}"
        if not mask_path.is_file():
            report.missing_masks.append(f"{defect}/{image_path.name}")
            continue

        image_size = _image_size(image_path)
        mask_size = _image_size(mask_path)
        if image_size != mask_size:
            report.size_mismatches.append(
                f"{defect}/{image_path.name}: image {image_size}, mask {mask_size}"
            )

        values = _mask_values(mask_path)
        report.mask_value_sets.add(tuple(sorted(values)))
        if not values <= EXPECTED_MASK_VALUES:
            report.non_binary_masks.append(f"{defect}/{mask_path.name}: values {sorted(values)}")

    expected = {f"{image_path.stem}{MASK_SUFFIX}" for image_path in images}
    report.orphan_masks.extend(
        f"{defect}/{mask_path.name}" for mask_path in masks if mask_path.name not in expected
    )


def build_reports(data_root: Path, categories: list[str] | None = None) -> list[CategoryReport]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"no dataset at {data_root}")

    names = categories or sorted(path.name for path in data_root.iterdir() if path.is_dir())
    return [build_category_report(data_root / name) for name in names]


def _format_sizes(sizes: Counter) -> str:
    return ", ".join(
        f"{width}x{height} ({count})" for (width, height), count in sizes.most_common()
    )


def _format_mask_values(value_sets: set[tuple[int, ...]]) -> str:
    if not value_sets:
        return "-"
    return "; ".join(str(list(values)) for values in sorted(value_sets))


def format_report(reports: list[CategoryReport], data_root: Path) -> str:
    lines = [
        "# Data sanity report",
        "",
        f"Data root: `{data_root.as_posix()}`",
        f"Categories audited: {len(reports)}",
        "",
    ]

    for report in reports:
        lines += [f"## {report.category}", ""]
        lines += ["| Split | Images | Masks |", "|---|---:|---:|"]
        lines.append(f"| `train/{GOOD_DIR}` | {report.train_good} | - |")
        for defect, count in report.test_counts.items():
            masks = "-" if defect == GOOD_DIR else str(report.mask_counts.get(defect, 0))
            lines.append(f"| `test/{defect}` | {count} | {masks} |")
        lines += [
            "",
            f"- Image sizes: {_format_sizes(report.image_sizes)}",
            f"- Mask value sets: {_format_mask_values(report.mask_value_sets)}",
            f"- Findings: {len(report.findings)}",
            "",
        ]

    all_findings = [finding for report in reports for finding in report.findings]
    lines += ["## Findings", ""]
    lines += [f"- {finding}" for finding in all_findings] or ["None."]
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the extracted MVTec AD dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec"))
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Audit only this category; repeatable. Defaults to every category found.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Also write Markdown here.")
    args = parser.parse_args()

    reports = build_reports(args.data_root, args.categories)
    rendered = format_report(reports, args.data_root)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Explicit newline so the report is byte-identical on Windows and Linux.
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"written to {args.output}")


if __name__ == "__main__":
    main()
