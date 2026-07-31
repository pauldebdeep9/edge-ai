"""Offline intrinsic camera calibration from a validated Task 4 manifest."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_ai.config import ProjectConfig

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
SUPPORTED_MANIFEST_VERSION = 1
PROJECT_RESOLUTION = (1280, 720)
PROJECT_PATTERN = (7, 7)
DEFAULT_CALIBRATION_FLAGS = 0
DEFAULT_TERMINATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-9,
)


class CalibrationError(RuntimeError):
    """Base error for intrinsic calibration."""


class ManifestValidationError(CalibrationError):
    """Raised when a Task 4 manifest is malformed or inconsistent."""


class InsufficientViewsError(CalibrationError):
    """Raised when too few accepted observations are available."""


class CalibrationNumericalError(CalibrationError):
    """Raised when OpenCV calibration fails or produces invalid values."""


@dataclass(frozen=True)
class CalibrationObservation:
    image_name: str
    image_points: Float32Array
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationDataset:
    source_manifest: Path
    resolution: tuple[int, int]
    pattern: tuple[int, int]
    square_size_mm: float
    observations: tuple[CalibrationObservation, ...]
    manifest_generated_at: str | None


@dataclass(frozen=True)
class PerImageReprojectionError:
    image_name: str
    rmse_pixels: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ReprojectionStatistics:
    mean_rmse_pixels: float
    median_rmse_pixels: float
    minimum_rmse_pixels: float
    maximum_rmse_pixels: float
    standard_deviation_pixels: float


@dataclass(frozen=True)
class CalibrationResult:
    opencv_rms: float
    camera_matrix: Float64Array
    distortion_coefficients: Float64Array
    rotation_vectors: tuple[Float64Array, ...]
    translation_vectors: tuple[Float64Array, ...]
    resolution: tuple[int, int]
    pattern: tuple[int, int]
    square_size_mm: float
    accepted_image_names: tuple[str, ...]
    accepted_image_warnings: tuple[tuple[str, ...], ...]
    calibration_flags: int
    termination_criteria: tuple[int, int, float]
    per_image_errors: tuple[PerImageReprojectionError, ...]
    error_statistics: ReprojectionStatistics
    software_versions: dict[str, str]
    source_manifest: Path
    manifest_generated_at: str | None
    calibration_timestamp: str
    advisory_observations: tuple[str, ...]

    @property
    def accepted_view_count(self) -> int:
        return len(self.accepted_image_names)


def generate_object_points(
    internal_corners_x: int,
    internal_corners_y: int,
    square_size_mm: float,
) -> NDArray[np.float32]:
    """Generate row-major planar checkerboard points in millimetres.

    The first internal corner is [0, 0, 0]. X increases across columns,
    Y increases across rows, and every Z coordinate is zero.
    """
    if (
        isinstance(internal_corners_x, bool)
        or not isinstance(internal_corners_x, int)
        or internal_corners_x <= 0
    ):
        raise ValueError("internal_corners_x must be a positive integer")
    if (
        isinstance(internal_corners_y, bool)
        or not isinstance(internal_corners_y, int)
        or internal_corners_y <= 0
    ):
        raise ValueError("internal_corners_y must be a positive integer")
    if (
        isinstance(square_size_mm, bool)
        or not isinstance(square_size_mm, (int, float))
        or not isfinite(float(square_size_mm))
        or square_size_mm <= 0
    ):
        raise ValueError("square_size_mm must be finite and positive")
    points = np.zeros(
        (internal_corners_x * internal_corners_y, 3), dtype=np.float32
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(internal_corners_x, dtype=np.float32),
        np.arange(internal_corners_y, dtype=np.float32),
    )
    points[:, 0] = grid_x.reshape(-1) * float(square_size_mm)
    points[:, 1] = grid_y.reshape(-1) * float(square_size_mm)
    return points


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ManifestValidationError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestValidationError(f"{name} must be a positive integer")
    return value


def load_calibration_dataset(
    manifest_path: Path,
    config: ProjectConfig,
    *,
    minimum_views: int = 10,
) -> CalibrationDataset:
    """Load and strictly validate accepted observations from a Task 4 manifest."""
    if (
        isinstance(minimum_views, bool)
        or not isinstance(minimum_views, int)
        or minimum_views <= 0
    ):
        raise ValueError("minimum_views must be a positive integer")
    resolved_manifest = manifest_path.expanduser().resolve()
    try:
        payload: object = json.loads(
            resolved_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"cannot load manifest {resolved_manifest}: {error}"
        ) from error
    root = _mapping(payload, "manifest")
    if root.get("manifest_version") != SUPPORTED_MANIFEST_VERSION:
        raise ManifestValidationError(
            f"unsupported manifest version: {root.get('manifest_version')!r}"
        )
    resolution = _mapping(
        root.get("expected_resolution"), "expected_resolution"
    )
    width = _positive_int(resolution.get("width"), "resolution.width")
    height = _positive_int(resolution.get("height"), "resolution.height")
    if (width, height) != PROJECT_RESOLUTION:
        raise ManifestValidationError(
            f"inconsistent resolution: expected 1280x720, found {width}x{height}"
        )
    if (width, height) != (config.camera.width, config.camera.height):
        raise ManifestValidationError(
            "manifest resolution does not match project configuration"
        )
    pattern_payload = _mapping(
        root.get("expected_internal_corner_pattern"),
        "expected_internal_corner_pattern",
    )
    pattern = (
        _positive_int(pattern_payload.get("x"), "pattern.x"),
        _positive_int(pattern_payload.get("y"), "pattern.y"),
    )
    if pattern != PROJECT_PATTERN or pattern != (
        config.checkerboard.internal_corners_x,
        config.checkerboard.internal_corners_y,
    ):
        raise ManifestValidationError(
            f"incorrect corner pattern: expected 7x7, found {pattern[0]}x{pattern[1]}"
        )
    square_size_mm = float(config.checkerboard.square_size_mm)
    if not isfinite(square_size_mm) or square_size_mm <= 0:
        raise ManifestValidationError(
            "square_size_mm must be finite and positive"
        )

    accepted_payload = root.get("accepted_images")
    if not isinstance(accepted_payload, list) or not all(
        isinstance(name, str) and name for name in accepted_payload
    ):
        raise ManifestValidationError(
            "accepted_images must be a list of non-empty strings"
        )
    accepted_names = cast(list[str], accepted_payload)
    if len(set(accepted_names)) != len(accepted_names):
        raise ManifestValidationError("accepted image identifiers must be unique")

    per_image_payload = root.get("per_image")
    if not isinstance(per_image_payload, list):
        raise ManifestValidationError("per_image must be a list")
    entries: dict[str, dict[str, object]] = {}
    explicitly_accepted: set[str] = set()
    for raw_entry in per_image_payload:
        entry = _mapping(raw_entry, "per_image entry")
        image_name = entry.get("image")
        if not isinstance(image_name, str) or not image_name:
            raise ManifestValidationError(
                "every per_image entry requires a non-empty image identifier"
            )
        if image_name in entries:
            raise ManifestValidationError(
                f"duplicate image identifier: {image_name}"
            )
        entries[image_name] = entry
        if entry.get("accepted") is True:
            explicitly_accepted.add(image_name)
    if set(accepted_names) != explicitly_accepted:
        raise ManifestValidationError(
            "accepted_images is inconsistent with explicitly accepted per_image entries"
        )
    if root.get("accepted_count") != len(accepted_names):
        raise ManifestValidationError(
            "accepted_count is inconsistent with accepted_images"
        )

    expected_point_count = pattern[0] * pattern[1]
    observations: list[CalibrationObservation] = []
    for image_name in sorted(accepted_names, key=lambda name: (name.casefold(), name)):
        entry = entries[image_name]
        if entry.get("width") != width or entry.get("height") != height:
            raise ManifestValidationError(
                f"inconsistent resolution for accepted image {image_name}"
            )
        raw_points = entry.get("refined_corners")
        if not isinstance(raw_points, list) or len(raw_points) != expected_point_count:
            raise ManifestValidationError(
                f"accepted image {image_name} must contain exactly "
                f"{expected_point_count} refined image points"
            )
        try:
            points = np.asarray(raw_points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"invalid coordinates for accepted image {image_name}"
            ) from error
        if points.shape != (expected_point_count, 2) or not np.isfinite(points).all():
            raise ManifestValidationError(
                f"invalid or non-finite coordinates for accepted image {image_name}"
            )
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= height)
        ):
            raise ManifestValidationError(
                f"image coordinates lie outside image bounds for {image_name}"
            )
        raw_warnings = entry.get("warnings", [])
        if not isinstance(raw_warnings, list) or not all(
            isinstance(warning, str) for warning in raw_warnings
        ):
            raise ManifestValidationError(
                f"warnings must be strings for accepted image {image_name}"
            )
        observations.append(
            CalibrationObservation(
                image_name=image_name,
                image_points=points.astype(np.float32),
                warnings=tuple(cast(list[str], raw_warnings)),
            )
        )
    if len(observations) < minimum_views:
        raise InsufficientViewsError(
            f"insufficient accepted images: {len(observations)} available, "
            f"minimum is {minimum_views}"
        )
    raw_manifest_generated_at = root.get("generated_at")
    manifest_generated_at = (
        raw_manifest_generated_at
        if isinstance(raw_manifest_generated_at, str)
        else None
    )
    return CalibrationDataset(
        source_manifest=resolved_manifest,
        resolution=(width, height),
        pattern=pattern,
        square_size_mm=square_size_mm,
        observations=tuple(observations),
        manifest_generated_at=manifest_generated_at,
    )


def _validate_result(
    rms: float,
    camera_matrix: Float64Array,
    distortion: Float64Array,
    rvecs: tuple[Float64Array, ...],
    tvecs: tuple[Float64Array, ...],
    view_count: int,
) -> None:
    if not isfinite(rms) or rms < 0:
        raise CalibrationNumericalError("OpenCV RMS is invalid")
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise CalibrationNumericalError("camera matrix must be finite and 3x3")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise CalibrationNumericalError("camera focal lengths must be positive")
    if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1e-6):
        raise CalibrationNumericalError(
            "camera matrix bottom row is not structurally plausible"
        )
    if distortion.size == 0 or not np.isfinite(distortion).all():
        raise CalibrationNumericalError(
            "distortion coefficients must be present and finite"
        )
    if len(rvecs) != view_count or len(tvecs) != view_count:
        raise CalibrationNumericalError(
            "rotation/translation vector counts do not match accepted views"
        )
    for vector in (*rvecs, *tvecs):
        if vector.size != 3 or not np.isfinite(vector).all():
            raise CalibrationNumericalError(
                "rotation and translation vectors must contain three finite values"
            )


def calibrate_intrinsics(
    dataset: CalibrationDataset,
    *,
    calibration_flags: int = DEFAULT_CALIBRATION_FLAGS,
    termination_criteria: tuple[int, int, float] = DEFAULT_TERMINATION_CRITERIA,
    calibrated_at: datetime | None = None,
) -> CalibrationResult:
    """Run one standard pinhole intrinsic calibration without removing views."""
    if not dataset.observations:
        raise InsufficientViewsError("no accepted calibration observations")
    object_template = generate_object_points(
        dataset.pattern[0], dataset.pattern[1], dataset.square_size_mm
    )
    object_points = [object_template.copy() for _ in dataset.observations]
    image_points = [
        observation.image_points.reshape(-1, 1, 2)
        for observation in dataset.observations
    ]
    if len(object_points) != len(image_points):
        raise ManifestValidationError(
            "image and object-point list lengths do not match"
        )
    try:
        (
            rms,
            camera_matrix,
            distortion,
            raw_rvecs,
            raw_tvecs,
        ) = cv2.calibrateCamera(
            object_points,
            image_points,
            dataset.resolution,
            None,
            None,
            flags=calibration_flags,
            criteria=termination_criteria,
        )
    except cv2.error as error:
        raise CalibrationNumericalError(
            f"OpenCV intrinsic calibration failed: {error}"
        ) from error
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
    rvecs = tuple(np.asarray(vector, dtype=np.float64).reshape(3, 1) for vector in raw_rvecs)
    tvecs = tuple(np.asarray(vector, dtype=np.float64).reshape(3, 1) for vector in raw_tvecs)
    _validate_result(
        float(rms), matrix, coefficients, rvecs, tvecs, len(dataset.observations)
    )

    per_image: list[PerImageReprojectionError] = []
    for observation, object_view, rvec, tvec in zip(
        dataset.observations, object_points, rvecs, tvecs, strict=True
    ):
        try:
            projected, _ = cv2.projectPoints(
                object_view, rvec, tvec, matrix, coefficients
            )
        except cv2.error as error:
            raise CalibrationNumericalError(
                f"reprojection failed for {observation.image_name}: {error}"
            ) from error
        projected_points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        observed_points = observation.image_points.astype(np.float64)
        squared_euclidean = np.sum(
            (projected_points - observed_points) ** 2, axis=1
        )
        rmse = float(np.sqrt(np.mean(squared_euclidean)))
        if not isfinite(rmse):
            raise CalibrationNumericalError(
                f"non-finite reprojection error for {observation.image_name}"
            )
        per_image.append(
            PerImageReprojectionError(
                image_name=observation.image_name,
                rmse_pixels=rmse,
                warnings=observation.warnings,
            )
        )
    error_values = np.asarray(
        [item.rmse_pixels for item in per_image], dtype=np.float64
    )
    statistics = ReprojectionStatistics(
        mean_rmse_pixels=float(np.mean(error_values)),
        median_rmse_pixels=float(np.median(error_values)),
        minimum_rmse_pixels=float(np.min(error_values)),
        maximum_rmse_pixels=float(np.max(error_values)),
        standard_deviation_pixels=float(np.std(error_values)),
    )
    advisory = [
        "No images were automatically removed; high-error views require human review."
    ]
    comparison_limit = (
        statistics.mean_rmse_pixels
        + 2.0 * statistics.standard_deviation_pixels
    )
    high_error_names = [
        item.image_name
        for item in per_image
        if item.rmse_pixels > comparison_limit
    ]
    if high_error_names:
        advisory.append(
            "Views above mean + 2 standard deviations (advisory only): "
            + ", ".join(high_error_names)
        )
    timestamp = calibrated_at or datetime.now(timezone.utc)
    return CalibrationResult(
        opencv_rms=float(rms),
        camera_matrix=matrix,
        distortion_coefficients=coefficients,
        rotation_vectors=rvecs,
        translation_vectors=tvecs,
        resolution=dataset.resolution,
        pattern=dataset.pattern,
        square_size_mm=dataset.square_size_mm,
        accepted_image_names=tuple(
            observation.image_name for observation in dataset.observations
        ),
        accepted_image_warnings=tuple(
            observation.warnings for observation in dataset.observations
        ),
        calibration_flags=calibration_flags,
        termination_criteria=termination_criteria,
        per_image_errors=tuple(per_image),
        error_statistics=statistics,
        software_versions={
            "python": platform.python_version(),
            "opencv": cv2.__version__,  # type: ignore[attr-defined]
            "numpy": np.__version__,
        },
        source_manifest=dataset.source_manifest,
        manifest_generated_at=dataset.manifest_generated_at,
        calibration_timestamp=timestamp.astimezone(timezone.utc).isoformat(),
        advisory_observations=tuple(advisory),
    )
