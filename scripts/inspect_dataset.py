"""Inspect calibration images offline and write annotations and a manifest."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from edge_ai.checkerboard import (  # noqa: E402
    SUPPORTED_IMAGE_EXTENSIONS,
    CheckerboardDetection,
    CheckerboardError,
    create_annotated_preview,
    detect_checkerboard,
)
from edge_ai.config import ConfigError, ProjectConfig, load_config  # noqa: E402


class DatasetInspectionError(RuntimeError):
    """Raised for offline dataset preparation or output failures."""


@dataclass(frozen=True)
class DatasetInspectionOptions:
    """Dataset-inspection command-line options."""

    config_path: Path
    input_directory: Path | None
    annotated_directory: Path | None
    manifest_path: Path | None
    extensions: frozenset[str]
    overwrite: bool


def _repository_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def _normalize_extensions(values: Sequence[str]) -> frozenset[str]:
    normalized = frozenset(
        value.casefold() if value.startswith(".") else f".{value.casefold()}"
        for value in values
    )
    unsupported = normalized - SUPPORTED_IMAGE_EXTENSIONS
    if unsupported:
        raise argparse.ArgumentTypeError(
            "unsupported image extensions: " + ", ".join(sorted(unsupported))
        )
    if not normalized:
        raise argparse.ArgumentTypeError("at least one extension is required")
    return normalized


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a calibration-image directory without a camera."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration.example.yaml"),
        help="configuration file, relative to the repository root",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="image directory; defaults to configured paths.raw_images",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        help="preview directory; defaults to configured paths.annotated_images",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path; defaults to output/dataset_manifest.json",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(SUPPORTED_IMAGE_EXTENSIONS),
        metavar="EXT",
        help="extensions to include, such as .png .jpg",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing manifest and annotated files",
    )
    return parser


def parse_options(
    argv: Sequence[str] | None = None,
) -> DatasetInspectionOptions:
    arguments = build_argument_parser().parse_args(argv)
    try:
        extensions = _normalize_extensions(arguments.extensions)
    except argparse.ArgumentTypeError as error:
        build_argument_parser().error(str(error))
    return DatasetInspectionOptions(
        config_path=arguments.config,
        input_directory=arguments.input_dir,
        annotated_directory=arguments.annotated_dir,
        manifest_path=arguments.manifest,
        extensions=extensions,
        overwrite=arguments.overwrite,
    )


def _resolve_outputs(
    options: DatasetInspectionOptions,
    config: ProjectConfig,
) -> tuple[Path, Path, Path]:
    input_directory = (
        config.paths.raw_images
        if options.input_directory is None
        else _repository_path(options.input_directory)
    )
    annotated_directory = (
        config.paths.annotated_images
        if options.annotated_directory is None
        else _repository_path(options.annotated_directory)
    )
    manifest_path = (
        config.paths.output / "dataset_manifest.json"
        if options.manifest_path is None
        else _repository_path(options.manifest_path)
    )
    return (
        input_directory.resolve(),
        annotated_directory.resolve(),
        manifest_path.resolve(),
    )


def enumerate_images(
    input_directory: Path, extensions: frozenset[str]
) -> tuple[Path, ...]:
    """Recursively enumerate image files in stable relative-path order."""
    candidates = (
        path
        for path in input_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    )
    return tuple(
        sorted(
            candidates,
            key=lambda path: (
                path.relative_to(input_directory).as_posix().casefold(),
                path.relative_to(input_directory).as_posix(),
            ),
        )
    )


def _coverage_summary(
    detections: Sequence[CheckerboardDetection],
) -> dict[str, object]:
    horizontal = [
        result.horizontal_coverage
        for result in detections
        if result.detection_success
        and result.horizontal_coverage is not None
    ]
    vertical = [
        result.vertical_coverage
        for result in detections
        if result.detection_success and result.vertical_coverage is not None
    ]

    def summarize(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "horizontal": summarize(horizontal),
        "vertical": summarize(vertical),
    }


def _configuration_manifest(
    config_path: Path, config: ProjectConfig
) -> dict[str, object]:
    return {
        "path": _repository_path(config_path).as_posix(),
        "camera": {
            "device_index": config.camera.device_index,
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps,
            "fourcc": config.camera.fourcc,
        },
        "checkerboard": {
            "printed_squares_x": config.checkerboard.printed_squares_x,
            "printed_squares_y": config.checkerboard.printed_squares_y,
            "internal_corners_x": config.checkerboard.internal_corners_x,
            "internal_corners_y": config.checkerboard.internal_corners_y,
            "square_size_mm": config.checkerboard.square_size_mm,
        },
    }


def inspect_dataset(
    options: DatasetInspectionOptions,
    *,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Inspect, annotate, and manifest an offline calibration dataset."""
    config = load_config(options.config_path)
    input_directory, annotated_directory, manifest_path = _resolve_outputs(
        options, config
    )
    if not input_directory.is_dir():
        raise DatasetInspectionError(
            f"input directory does not exist: {input_directory}"
        )
    if annotated_directory == input_directory or input_directory in (
        annotated_directory.parents
    ):
        raise DatasetInspectionError(
            "annotated directory must be outside the source image directory"
        )
    if manifest_path.exists() and not options.overwrite:
        raise DatasetInspectionError(
            f"refusing to overwrite existing manifest: {manifest_path}"
        )
    if (
        annotated_directory.exists()
        and any(annotated_directory.rglob("*"))
        and not options.overwrite
    ):
        raise DatasetInspectionError(
            "refusing to overwrite existing annotated dataset: "
            f"{annotated_directory}"
        )

    image_paths = enumerate_images(input_directory, options.extensions)
    try:
        annotated_directory.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DatasetInspectionError(
            f"cannot create dataset output directory: {error}"
        ) from error

    detections: list[CheckerboardDetection] = []
    per_image: list[dict[str, object]] = []
    accepted_images: list[str] = []
    rejected_images: list[dict[str, str]] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(input_directory)
        detection = detect_checkerboard(
            image_path,
            config.checkerboard,
            expected_width=config.camera.width,
            expected_height=config.camera.height,
            supported_extensions=options.extensions,
        )
        detections.append(detection)
        annotation = create_annotated_preview(
            image_path,
            detection,
            config.checkerboard,
            fallback_width=config.camera.width,
            fallback_height=config.camera.height,
        )
        annotation_relative_path = relative_path.with_name(
            f"{relative_path.stem}.annotated.png"
        )
        annotation_path = annotated_directory / annotation_relative_path
        try:
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DatasetInspectionError(
                f"cannot create annotation directory: {error}"
            ) from error
        if annotation_path.exists() and not options.overwrite:
            raise DatasetInspectionError(
                f"refusing to overwrite annotated image: {annotation_path}"
            )
        if not cv2.imwrite(str(annotation_path), annotation):
            raise DatasetInspectionError(
                f"cannot write annotated image: {annotation_path}"
            )

        identifier = relative_path.as_posix()
        result_payload = detection.to_dict(image_identifier=identifier)
        result_payload["annotated_image"] = (
            annotation_relative_path.as_posix()
        )
        per_image.append(result_payload)
        if detection.detection_success:
            accepted_images.append(identifier)
        else:
            rejected_images.append(
                {
                    "image": identifier,
                    "reason": detection.rejection_reason
                    or "unspecified rejection",
                }
            )

    accepted_count = len(accepted_images)
    rejected_count = len(rejected_images)
    warning_count = sum(len(result.warnings) for result in detections)
    timestamp = generated_at or datetime.now(timezone.utc)
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "generated_at": timestamp.astimezone(timezone.utc).isoformat(),
        "configuration": _configuration_manifest(options.config_path, config),
        "input_directory": input_directory.as_posix(),
        "annotated_directory": annotated_directory.as_posix(),
        "expected_resolution": {
            "width": config.camera.width,
            "height": config.camera.height,
        },
        "expected_internal_corner_pattern": {
            "x": config.checkerboard.internal_corners_x,
            "y": config.checkerboard.internal_corners_y,
        },
        "extensions": sorted(options.extensions),
        "total_image_count": len(image_paths),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "warning_count": warning_count,
        "per_image": per_image,
        "accepted_images": accepted_images,
        "rejected_images": rejected_images,
        "coverage_summary": _coverage_summary(detections),
        "software_versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,  # type: ignore[attr-defined]
            "numpy": np.__version__,
        },
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise DatasetInspectionError(
            f"cannot write dataset manifest {manifest_path}: {error}"
        ) from error
    return manifest


def run_inspection(
    options: DatasetInspectionOptions,
    *,
    generated_at: datetime | None = None,
) -> int:
    manifest = inspect_dataset(options, generated_at=generated_at)
    print(
        "Dataset inspection: "
        f"{manifest['total_image_count']} total, "
        f"{manifest['accepted_count']} accepted, "
        f"{manifest['rejected_count']} rejected, "
        f"{manifest['warning_count']} warnings"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_options(argv)
    try:
        return run_inspection(options)
    except (
        CheckerboardError,
        ConfigError,
        DatasetInspectionError,
        OSError,
        ValueError,
    ) as error:
        print(f"Dataset inspection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
