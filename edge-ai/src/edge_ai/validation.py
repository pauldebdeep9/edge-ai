"""Offline image undistortion and calibration-validation reporting."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_ai.storage import LoadedCalibration, load_calibration

ImageArray = NDArray[np.generic]
Float64Array = NDArray[np.float64]
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)


class ValidationError(RuntimeError):
    """Base error for offline calibration validation."""


class ImageValidationError(ValidationError):
    """Raised when a source image cannot be safely validated."""


class ResolutionMismatchError(ValidationError):
    """Raised when image and calibration resolutions differ."""


class UndistortionError(ValidationError):
    """Raised when OpenCV cannot produce a valid full-frame result."""


class ValidationStorageError(ValidationError):
    """Raised when a complete validation artifact set cannot be saved."""


@dataclass(frozen=True)
class LoadedImage:
    """Validated source image and its resolved location."""

    source_path: Path
    pixels: ImageArray
    width: int
    height: int

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class UndistortionResult:
    """Undistorted images and geometry returned by the pinhole model."""

    full_image: ImageArray
    cropped_image: ImageArray | None
    comparison_image: ImageArray
    new_camera_matrix: Float64Array
    opencv_roi: tuple[int, int, int, int]
    valid_roi: tuple[int, int, int, int] | None
    crop_requested: bool
    crop_omission_reason: str | None
    alpha: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ValidationOutputPaths:
    """Final paths for one complete validation artifact set."""

    full: Path
    cropped: Path | None
    comparison: Path
    report: Path


@dataclass(frozen=True)
class ValidationRun:
    """Complete in-memory and persisted result of one validation run."""

    calibration: LoadedCalibration
    source_image: LoadedImage
    undistortion: UndistortionResult
    output_paths: ValidationOutputPaths
    report: dict[str, object]


@dataclass(frozen=True)
class _CandidateOutputPaths:
    full: Path
    cropped: Path
    comparison: Path
    report: Path


ImageReader = Callable[[str, int], object]
PngWriter = Callable[[Path, ImageArray], None]
JsonWriter = Callable[[Path, dict[str, object]], None]


def load_validation_image(
    image_path: Path,
    *,
    image_reader: ImageReader = cv2.imread,
) -> LoadedImage:
    """Load an image without modifying, resizing, or recompressing its source."""
    path = image_path.expanduser().resolve()
    if not path.exists():
        raise ImageValidationError(f"image file does not exist: {path}")
    if not path.is_file():
        raise ImageValidationError(f"image path is not a file: {path}")
    if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ImageValidationError(
            f"unsupported image extension {path.suffix!r}; expected {supported}"
        )
    try:
        loaded = image_reader(str(path), cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error) as error:
        raise ImageValidationError(
            f"cannot read source image {path}: {error}"
        ) from error
    if not isinstance(loaded, np.ndarray):
        raise ImageValidationError(f"source image is unreadable: {path}")
    pixels = cast(ImageArray, loaded)
    if pixels.size == 0:
        raise ImageValidationError(f"source image is empty: {path}")
    if pixels.ndim not in {2, 3}:
        raise ImageValidationError(
            f"source image has unsupported array shape: {pixels.shape}"
        )
    if pixels.ndim == 3 and pixels.shape[2] not in {1, 3, 4}:
        raise ImageValidationError(
            f"source image has unsupported channel count: {pixels.shape[2]}"
        )
    if pixels.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise ImageValidationError(
            "source image pixels must be 8-bit or 16-bit unsigned integers"
        )
    height, width = pixels.shape[:2]
    if width <= 0 or height <= 0:
        raise ImageValidationError(
            f"source image dimensions are invalid: {width}x{height}"
        )
    return LoadedImage(
        source_path=path,
        pixels=pixels,
        width=int(width),
        height=int(height),
    )


def require_matching_resolution(
    source_image: LoadedImage, calibration: LoadedCalibration
) -> None:
    """Require exact dimensions because stored intrinsics are resolution-dependent."""
    if source_image.resolution != calibration.resolution:
        raise ResolutionMismatchError(
            "input image resolution "
            f"{source_image.width}x{source_image.height} does not match "
            "calibration resolution "
            f"{calibration.image_width}x{calibration.image_height}; camera "
            "intrinsics are resolution-dependent, and Task 6 does not resize "
            "images or scale camera matrices"
        )


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise UndistortionError("alpha must be a number from 0.0 to 1.0")
    number = float(alpha)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise UndistortionError(
            "alpha must be finite and within the inclusive range 0.0 to 1.0"
        )
    return number


def _roi_tuple(value: object) -> tuple[int, int, int, int]:
    try:
        raw_values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise UndistortionError(
            f"OpenCV returned an invalid ROI value: {value!r}"
        ) from error
    if len(raw_values) != 4:
        raise UndistortionError(
            f"OpenCV returned an invalid ROI value: {value!r}"
        )
    converted: list[int] = []
    for item in raw_values:
        if isinstance(item, bool) or not isinstance(
            item, (int, np.integer)
        ):
            raise UndistortionError(
                f"OpenCV returned a non-integer ROI value: {value!r}"
            )
        converted.append(int(item))
    return converted[0], converted[1], converted[2], converted[3]


def _valid_roi(
    roi: tuple[int, int, int, int], width: int, height: int
) -> bool:
    x, y, roi_width, roi_height = roi
    return (
        x >= 0
        and y >= 0
        and roi_width > 0
        and roi_height > 0
        and x + roi_width <= width
        and y + roi_height <= height
    )


def _comparison_panel(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        converted = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        converted = image.copy()
    return cast(ImageArray, converted)


def _create_comparison(
    original: ImageArray, undistorted: ImageArray
) -> ImageArray:
    original_panel = _comparison_panel(original)
    undistorted_panel = _comparison_panel(undistorted)
    combined = np.concatenate(
        (original_panel, undistorted_panel), axis=1
    )
    comparison = cv2.copyMakeBorder(
        combined,
        40,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    maximum = int(np.iinfo(comparison.dtype).max)
    text_color = (maximum, maximum, maximum)
    cv2.putText(
        comparison,
        "ORIGINAL",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        "UNDISTORTED",
        (original_panel.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return cast(ImageArray, comparison)


def undistort_image(
    source_image: LoadedImage,
    calibration: LoadedCalibration,
    *,
    alpha: float = 0.0,
    crop: bool = True,
) -> UndistortionResult:
    """Undistort one exact-resolution image with the standard pinhole model.

    Alpha 0 generally prioritizes valid pixels and more cropping. Alpha 1
    generally retains more field of view and can preserve black borders.
    """
    validated_alpha = _validate_alpha(alpha)
    require_matching_resolution(source_image, calibration)
    image_size = source_image.resolution
    try:
        raw_new_matrix, raw_roi = cv2.getOptimalNewCameraMatrix(
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            image_size,
            validated_alpha,
            image_size,
        )
        raw_full_image = cv2.undistort(
            source_image.pixels,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            raw_new_matrix,
        )
    except cv2.error as error:
        raise UndistortionError(f"OpenCV undistortion failed: {error}") from error
    new_camera_matrix = np.asarray(raw_new_matrix, dtype=np.float64)
    if (
        new_camera_matrix.shape != (3, 3)
        or not np.isfinite(new_camera_matrix).all()
    ):
        raise UndistortionError(
            "OpenCV returned an invalid new camera matrix"
        )
    if (
        new_camera_matrix[0, 0] <= 0
        or new_camera_matrix[1, 1] <= 0
    ):
        raise UndistortionError(
            "OpenCV returned non-positive new focal lengths"
        )
    if not isinstance(raw_full_image, np.ndarray):
        raise UndistortionError("OpenCV returned no undistorted image")
    full_image = cast(ImageArray, raw_full_image)
    if full_image.size == 0 or full_image.shape != source_image.pixels.shape:
        raise UndistortionError(
            "full undistorted image must be non-empty and preserve "
            "the source dimensions"
        )
    if not np.isfinite(full_image).all():
        raise UndistortionError(
            "full undistorted image contains non-finite pixels"
        )

    roi = _roi_tuple(raw_roi)
    warnings: list[str] = []
    cropped_image: ImageArray | None = None
    valid_roi: tuple[int, int, int, int] | None = None
    omission_reason: str | None = None
    if _valid_roi(roi, source_image.width, source_image.height):
        valid_roi = roi
        if crop:
            x, y, roi_width, roi_height = roi
            cropped_image = full_image[
                y : y + roi_height, x : x + roi_width
            ].copy()
            if cropped_image.size == 0:
                raise UndistortionError(
                    "valid ROI unexpectedly produced an empty crop"
                )
        else:
            omission_reason = "cropping disabled by request"
    else:
        omission_reason = (
            "OpenCV returned an empty or invalid ROI; full-frame output "
            "remains available"
        )
        warnings.append(omission_reason)

    comparison = _create_comparison(
        source_image.pixels, full_image
    )
    if comparison.size == 0 or not np.isfinite(comparison).all():
        raise UndistortionError(
            "comparison image is empty or contains invalid pixels"
        )
    return UndistortionResult(
        full_image=full_image,
        cropped_image=cropped_image,
        comparison_image=comparison,
        new_camera_matrix=new_camera_matrix,
        opencv_roi=roi,
        valid_roi=valid_roi,
        crop_requested=crop,
        crop_omission_reason=omission_reason,
        alpha=validated_alpha,
        warnings=tuple(warnings),
    )


def _candidate_output_paths(
    output_directory: Path, output_prefix: str
) -> _CandidateOutputPaths:
    if (
        not output_prefix
        or Path(output_prefix).name != output_prefix
        or output_prefix in {".", ".."}
    ):
        raise ValidationStorageError(
            "output_prefix must be a non-empty filename prefix"
        )
    directory = output_directory.expanduser().resolve()
    return _CandidateOutputPaths(
        full=directory / f"{output_prefix}_undistorted_full.png",
        cropped=directory / f"{output_prefix}_undistorted_cropped.png",
        comparison=directory / f"{output_prefix}_comparison.png",
        report=directory / f"{output_prefix}_report.json",
    )


def _dimensions(image: ImageArray | None) -> dict[str, int] | None:
    if image is None:
        return None
    return {"width": int(image.shape[1]), "height": int(image.shape[0])}


def _roi_payload(
    roi: tuple[int, int, int, int] | None
) -> dict[str, int] | None:
    if roi is None:
        return None
    return {
        "x": roi[0],
        "y": roi[1],
        "width": roi[2],
        "height": roi[3],
    }


def validation_report_payload(
    calibration: LoadedCalibration,
    source_image: LoadedImage,
    result: UndistortionResult,
    output_paths: ValidationOutputPaths,
    *,
    validated_at: datetime,
) -> dict[str, object]:
    """Build a deterministic JSON-compatible Task 6 validation report."""
    return {
        "report_type": "offline intrinsic-calibration undistortion validation",
        "validation_timestamp": (
            validated_at.astimezone(timezone.utc).isoformat()
        ),
        "source_image_path": source_image.source_path.as_posix(),
        "calibration_path": calibration.source_path.as_posix(),
        "calibration_source_format": calibration.source_format,
        "original_resolution": {
            "width": source_image.width,
            "height": source_image.height,
        },
        "calibration_resolution": {
            "width": calibration.image_width,
            "height": calibration.image_height,
        },
        "resolution_match": (
            source_image.resolution == calibration.resolution
        ),
        "explicit_scaling_used": False,
        "alpha": result.alpha,
        "alpha_explanation": (
            "Alpha 0 generally prioritizes valid pixels and more cropping; "
            "alpha 1 generally retains more field of view and may preserve "
            "black borders."
        ),
        "original_camera_matrix": calibration.camera_matrix.tolist(),
        "new_camera_matrix": result.new_camera_matrix.tolist(),
        "distortion_coefficients": (
            calibration.distortion_coefficients.tolist()
        ),
        "opencv_roi": _roi_payload(result.opencv_roi),
        "valid_roi": _roi_payload(result.valid_roi),
        "crop_requested": result.crop_requested,
        "crop_omission_reason": result.crop_omission_reason,
        "full_output_path": output_paths.full.as_posix(),
        "cropped_output_path": (
            output_paths.cropped.as_posix()
            if output_paths.cropped is not None
            else None
        ),
        "comparison_output_path": output_paths.comparison.as_posix(),
        "report_output_path": output_paths.report.as_posix(),
        "output_dimensions": {
            "full": _dimensions(result.full_image),
            "cropped": _dimensions(result.cropped_image),
            "comparison": _dimensions(result.comparison_image),
        },
        "calibration_metadata": {
            "internal_corners_x": calibration.internal_corners_x,
            "internal_corners_y": calibration.internal_corners_y,
            "square_size_mm": calibration.square_size_mm,
            "opencv_rms": calibration.opencv_rms,
            "mean_reprojection_rmse_pixels": (
                calibration.mean_reprojection_rmse_pixels
            ),
        },
        "checkerboard_observations": None,
        "warnings": list(result.warnings),
        "software_versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,  # type: ignore[attr-defined]
            "numpy": np.__version__,
        },
        "visual_inspection_required": (
            "Visual inspection is still required; a plausible image does "
            "not prove calibration accuracy."
        ),
        "limitations": (
            "Undistortion from camera intrinsics does not establish "
            "arbitrary 3D object position or physical distance."
        ),
    }


def _write_png(path: Path, image: ImageArray) -> None:
    success, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not success or encoded.size == 0:
        raise ValidationStorageError(f"cannot encode lossless PNG: {path}")
    with path.open("wb") as output:
        output.write(encoded.tobytes())
        output.flush()
        os.fsync(output.fileno())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _validate_png(path: Path, expected: ImageArray) -> None:
    reloaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        reloaded is None
        or reloaded.shape != expected.shape
        or reloaded.dtype != expected.dtype
        or not np.array_equal(reloaded, expected)
    ):
        raise ValidationStorageError(
            f"generated PNG failed lossless read-back validation: {path}"
        )


def _validate_json(path: Path) -> None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationStorageError(
            f"generated JSON failed read-back validation: {path}"
        ) from error
    if not isinstance(loaded, dict):
        raise ValidationStorageError(
            f"generated validation report is not a JSON object: {path}"
        )


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".validation-", suffix=suffix, dir=directory
    )
    os.close(descriptor)
    return Path(name)


def save_validation_result(
    calibration: LoadedCalibration,
    source_image: LoadedImage,
    result: UndistortionResult,
    output_directory: Path,
    *,
    output_prefix: str = "calibration_validation",
    overwrite: bool = False,
    validated_at: datetime | None = None,
    png_writer: PngWriter = _write_png,
    json_writer: JsonWriter = _write_json,
) -> tuple[ValidationOutputPaths, dict[str, object]]:
    """Stage, verify, and atomically replace a rollback-capable output set."""
    candidates = _candidate_output_paths(output_directory, output_prefix)
    potential_targets = (
        candidates.full,
        candidates.cropped,
        candidates.comparison,
        candidates.report,
    )
    source_path = source_image.source_path.resolve()
    if source_path in potential_targets:
        raise ValidationStorageError(
            "validation output would overwrite the original source image"
        )
    existing = [path for path in potential_targets if path.exists()]
    if existing and not overwrite:
        raise ValidationStorageError(
            "refusing to overwrite validation outputs: "
            + ", ".join(str(path) for path in existing)
        )
    try:
        candidates.full.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValidationStorageError(
            f"cannot create validation output directory: {error}"
        ) from error

    output_paths = ValidationOutputPaths(
        full=candidates.full,
        cropped=(
            candidates.cropped
            if result.cropped_image is not None
            else None
        ),
        comparison=candidates.comparison,
        report=candidates.report,
    )
    timestamp = validated_at or datetime.now(timezone.utc)
    report = validation_report_payload(
        calibration,
        source_image,
        result,
        output_paths,
        validated_at=timestamp,
    )
    images: tuple[tuple[Path, ImageArray], ...] = (
        (candidates.full, result.full_image),
        *(
            ((candidates.cropped, result.cropped_image),)
            if result.cropped_image is not None
            else ()
        ),
        (candidates.comparison, result.comparison_image),
    )
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target, image in images:
            temporary = _temporary_path(
                target.parent, f".{target.stem}.png"
            )
            staged[target] = temporary
            png_writer(temporary, image)
            _validate_png(temporary, image)
        report_temporary = _temporary_path(
            candidates.report.parent, ".report.json"
        )
        staged[candidates.report] = report_temporary
        json_writer(report_temporary, report)
        _validate_json(report_temporary)

        if overwrite:
            for target in potential_targets:
                if target.exists():
                    backup = _temporary_path(
                        target.parent, f".backup{target.suffix}"
                    )
                    backup.unlink()
                    target.replace(backup)
                    backups[target] = backup
        generated_targets = tuple(target for target, _ in images) + (
            candidates.report,
        )
        for target in generated_targets:
            staged[target].replace(target)
            committed.append(target)
    except Exception as error:
        for target in committed:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                backup.replace(target)
        if isinstance(error, ValidationStorageError):
            raise
        raise ValidationStorageError(
            f"failed to save complete validation output set: {error}"
        ) from error
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    return output_paths, report


def validate_calibration_image(
    calibration_path: Path,
    image_path: Path,
    output_directory: Path,
    *,
    output_prefix: str = "calibration_validation",
    alpha: float = 0.0,
    crop: bool = True,
    overwrite: bool = False,
    validated_at: datetime | None = None,
) -> ValidationRun:
    """Load, undistort, report, and persist one offline validation image."""
    calibration = load_calibration(calibration_path)
    source_image = load_validation_image(image_path)
    result = undistort_image(
        source_image, calibration, alpha=alpha, crop=crop
    )
    output_paths, report = save_validation_result(
        calibration,
        source_image,
        result,
        output_directory,
        output_prefix=output_prefix,
        overwrite=overwrite,
        validated_at=validated_at,
    )
    return ValidationRun(
        calibration=calibration,
        source_image=source_image,
        undistortion=result,
        output_paths=output_paths,
        report=report,
    )
