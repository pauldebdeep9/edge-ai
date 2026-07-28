"""Validate Task 5 intrinsics by undistorting one offline source image."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from edge_ai.storage import CalibrationLoadError  # noqa: E402
from edge_ai.validation import (  # noqa: E402
    ValidationError,
    ValidationRun,
    validate_calibration_image,
)


@dataclass(frozen=True)
class ValidationOptions:
    calibration_path: Path
    image_path: Path
    output_directory: Path
    output_prefix: str
    alpha: float
    crop: bool
    overwrite: bool


def _repository_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Task 5 calibration by undistorting an offline image."
        )
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Task 5 .npz, .yaml, or .yml calibration file",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="offline source image captured at the calibration resolution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/validation"),
        help="validation output directory (default: output/validation)",
    )
    parser.add_argument(
        "--output-prefix",
        default="calibration_validation",
        help="shared prefix for PNG and JSON output names",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "0.0 favors valid pixels/cropping; 1.0 retains more field of "
            "view and may retain black borders (default: 0.0)"
        ),
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop",
        dest="crop",
        action="store_true",
        help="write the valid-ROI crop when available (default)",
    )
    crop_group.add_argument(
        "--no-crop",
        dest="crop",
        action="store_false",
        help="omit the valid-ROI cropped output",
    )
    parser.set_defaults(crop=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing complete validation output set",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> ValidationOptions:
    arguments = build_argument_parser().parse_args(argv)
    return ValidationOptions(
        calibration_path=_repository_path(arguments.calibration),
        image_path=_repository_path(arguments.image),
        output_directory=_repository_path(arguments.output_dir),
        output_prefix=arguments.output_prefix,
        alpha=arguments.alpha,
        crop=arguments.crop,
        overwrite=arguments.overwrite,
    )


def _print_summary(run: ValidationRun) -> None:
    calibration = run.calibration
    source = run.source_image
    result = run.undistortion
    paths = run.output_paths
    original = calibration.camera_matrix
    updated = result.new_camera_matrix
    print(
        f"Calibration: {calibration.source_path} "
        f"({calibration.source_format})"
    )
    print(f"Source image: {source.source_path}")
    print(
        "Source/calibration resolution: "
        f"{source.width}x{source.height} / "
        f"{calibration.image_width}x{calibration.image_height}"
    )
    print(f"Alpha: {result.alpha:g}")
    print(
        "Original fx/fy/cx/cy: "
        f"{original[0, 0]:.6f} / {original[1, 1]:.6f} / "
        f"{original[0, 2]:.6f} / {original[1, 2]:.6f}"
    )
    print(
        "New fx/fy/cx/cy: "
        f"{updated[0, 0]:.6f} / {updated[1, 1]:.6f} / "
        f"{updated[0, 2]:.6f} / {updated[1, 2]:.6f}"
    )
    print(
        "OpenCV ROI (x, y, width, height): "
        f"{result.opencv_roi}"
    )
    print(
        "Full output dimensions: "
        f"{result.full_image.shape[1]}x{result.full_image.shape[0]}"
    )
    if result.cropped_image is not None:
        print(
            "Cropped output dimensions: "
            f"{result.cropped_image.shape[1]}x"
            f"{result.cropped_image.shape[0]}"
        )
    else:
        print(
            "Cropped output omitted: "
            f"{result.crop_omission_reason or 'not generated'}"
        )
    print(f"Full PNG: {paths.full}")
    if paths.cropped is not None:
        print(f"Cropped PNG: {paths.cropped}")
    print(f"Comparison PNG: {paths.comparison}")
    print(f"JSON report: {paths.report}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def run_validation(
    options: ValidationOptions,
    *,
    validated_at: datetime | None = None,
) -> int:
    run = validate_calibration_image(
        options.calibration_path,
        options.image_path,
        options.output_directory,
        output_prefix=options.output_prefix,
        alpha=options.alpha,
        crop=options.crop,
        overwrite=options.overwrite,
        validated_at=validated_at,
    )
    _print_summary(run)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_options(argv)
    try:
        return run_validation(options)
    except (CalibrationLoadError, ValidationError, OSError, ValueError) as error:
        print(f"Calibration validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
