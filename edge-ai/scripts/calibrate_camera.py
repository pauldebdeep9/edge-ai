"""Calibrate camera intrinsics offline from a Task 4 dataset manifest."""

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

from edge_ai.calibration import (  # noqa: E402
    CalibrationError,
    calibrate_intrinsics,
    load_calibration_dataset,
)
from edge_ai.config import ConfigError, load_config  # noqa: E402
from edge_ai.storage import (  # noqa: E402
    CalibrationOutputPaths,
    CalibrationStorageError,
    save_calibration_result,
)


@dataclass(frozen=True)
class CalibrationOptions:
    config_path: Path
    manifest_path: Path | None
    output_directory: Path | None
    output_prefix: str
    minimum_views: int
    overwrite: bool


def _positive_integer(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return integer


def _repository_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline intrinsic checkerboard calibration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration.example.yaml"),
        help="configuration file, relative to the repository root",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Task 4 manifest; defaults to output/dataset_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; defaults to configured paths.output",
    )
    parser.add_argument(
        "--output-prefix",
        default="camera_calibration",
        help="shared NPZ/YAML/JSON filename prefix",
    )
    parser.add_argument(
        "--minimum-views",
        type=_positive_integer,
        default=10,
        help="minimum accepted views required (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing complete output set",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> CalibrationOptions:
    arguments = build_argument_parser().parse_args(argv)
    return CalibrationOptions(
        config_path=arguments.config,
        manifest_path=arguments.manifest,
        output_directory=arguments.output_dir,
        output_prefix=arguments.output_prefix,
        minimum_views=arguments.minimum_views,
        overwrite=arguments.overwrite,
    )


def _print_summary(
    result: object, paths: CalibrationOutputPaths
) -> None:
    from edge_ai.calibration import CalibrationResult

    if not isinstance(result, CalibrationResult):
        raise TypeError("unexpected calibration result")
    matrix = result.camera_matrix
    statistics = result.error_statistics
    print(f"Accepted images used: {result.accepted_view_count}")
    print(f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Checkerboard pattern: {result.pattern[0]}x{result.pattern[1]}")
    print(f"Square size: {result.square_size_mm:g} mm")
    print(f"fx/fy: {matrix[0, 0]:.6f} / {matrix[1, 1]:.6f}")
    print(f"cx/cy: {matrix[0, 2]:.6f} / {matrix[1, 2]:.6f}")
    print(
        "Distortion coefficient count: "
        f"{result.distortion_coefficients.size}"
    )
    print(f"OpenCV RMS: {result.opencv_rms:.6f}")
    print(
        "Per-image RMSE mean/median/worst: "
        f"{statistics.mean_rmse_pixels:.6f} / "
        f"{statistics.median_rmse_pixels:.6f} / "
        f"{statistics.maximum_rmse_pixels:.6f} pixels"
    )
    print(f"NPZ: {paths.npz}")
    print(f"OpenCV YAML: {paths.yaml}")
    print(f"JSON: {paths.json}")


def run_calibration(
    options: CalibrationOptions,
    *,
    calibrated_at: datetime | None = None,
) -> int:
    config = load_config(options.config_path)
    manifest_path = (
        config.paths.output / "dataset_manifest.json"
        if options.manifest_path is None
        else _repository_path(options.manifest_path)
    )
    output_directory = (
        config.paths.output
        if options.output_directory is None
        else _repository_path(options.output_directory)
    )
    dataset = load_calibration_dataset(
        manifest_path, config, minimum_views=options.minimum_views
    )
    result = calibrate_intrinsics(dataset, calibrated_at=calibrated_at)
    paths = save_calibration_result(
        result,
        output_directory,
        output_prefix=options.output_prefix,
        overwrite=options.overwrite,
    )
    _print_summary(result, paths)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_options(argv)
    try:
        return run_calibration(options)
    except (
        CalibrationError,
        CalibrationStorageError,
        ConfigError,
        OSError,
        ValueError,
    ) as error:
        print(f"Intrinsic calibration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
