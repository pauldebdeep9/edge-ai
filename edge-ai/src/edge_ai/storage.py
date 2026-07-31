"""Atomic persistence for intrinsic calibration results."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_ai.calibration import CalibrationResult

Float64Array = NDArray[np.float64]
CalibrationSourceFormat = Literal["npz", "opencv-yaml"]


class CalibrationStorageError(RuntimeError):
    """Raised when a complete calibration output set cannot be persisted."""


class CalibrationLoadError(RuntimeError):
    """Raised when persisted calibration parameters cannot be loaded safely."""


@dataclass(frozen=True)
class CalibrationOutputPaths:
    npz: Path
    yaml: Path
    json: Path


@dataclass(frozen=True)
class LoadedCalibration:
    """Validated standard-pinhole calibration loaded from Task 5 output."""

    camera_matrix: Float64Array
    distortion_coefficients: Float64Array
    image_width: int
    image_height: int
    internal_corners_x: int | None
    internal_corners_y: int | None
    square_size_mm: float | None
    opencv_rms: float | None
    mean_reprojection_rmse_pixels: float | None
    source_path: Path
    source_format: CalibrationSourceFormat

    def __post_init__(self) -> None:
        try:
            matrix = np.asarray(self.camera_matrix, dtype=np.float64)
            distortion = np.asarray(
                self.distortion_coefficients, dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as error:
            raise CalibrationLoadError(
                "camera matrix and distortion coefficients must be numeric"
            ) from error
        if matrix.shape != (3, 3):
            raise CalibrationLoadError(
                "camera matrix shape must be exactly 3x3"
            )
        if not np.isfinite(matrix).all():
            raise CalibrationLoadError("camera matrix values must be finite")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise CalibrationLoadError("camera focal lengths must be positive")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise CalibrationLoadError(
                "camera matrix bottom row is not structurally plausible"
            )
        if distortion.size == 0:
            raise CalibrationLoadError(
                "distortion coefficients are missing or empty"
            )
        if not np.isfinite(distortion).all():
            raise CalibrationLoadError(
                "distortion coefficient values must be finite"
            )
        _validate_positive_integer(self.image_width, "image_width")
        _validate_positive_integer(self.image_height, "image_height")
        if (self.internal_corners_x is None) != (
            self.internal_corners_y is None
        ):
            raise CalibrationLoadError(
                "checkerboard dimensions must both be present or both be absent"
            )
        if self.internal_corners_x is not None:
            _validate_positive_integer(
                self.internal_corners_x, "internal_corners_x"
            )
            _validate_positive_integer(
                self.internal_corners_y, "internal_corners_y"
            )
        if self.square_size_mm is not None and (
            not isfinite(self.square_size_mm) or self.square_size_mm <= 0
        ):
            raise CalibrationLoadError(
                "square_size_mm must be finite and positive when present"
            )
        for value, name in (
            (self.opencv_rms, "OpenCV RMS"),
            (
                self.mean_reprojection_rmse_pixels,
                "mean reprojection RMSE",
            ),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise CalibrationLoadError(
                    f"{name} must be finite and non-negative when present"
                )
        if self.source_format not in {"npz", "opencv-yaml"}:
            raise CalibrationLoadError(
                f"unsupported calibration source format: {self.source_format}"
            )
        object.__setattr__(self, "camera_matrix", matrix.copy())
        object.__setattr__(
            self, "distortion_coefficients", distortion.copy()
        )
        object.__setattr__(
            self, "source_path", self.source_path.expanduser().resolve()
        )

    @property
    def resolution(self) -> tuple[int, int]:
        return self.image_width, self.image_height


Writer = Callable[[Path, CalibrationResult], None]


def _validate_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CalibrationLoadError(f"{name} must be a positive integer")
    if int(value) <= 0:
        raise CalibrationLoadError(f"{name} must be a positive integer")


def _number_as_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CalibrationLoadError(f"{name} must be a positive integer")
    number = float(value)
    if not isfinite(number) or not number.is_integer() or number <= 0:
        raise CalibrationLoadError(f"{name} must be a positive integer")
    return int(number)


def _optional_finite_float(
    value: object | None,
    name: str,
    *,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CalibrationLoadError(f"{name} must be numeric when present")
    number = float(value)
    if not isfinite(number) or (positive and number <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise CalibrationLoadError(
            f"{name} must be {qualifier} when present"
        )
    return number


def _npz_required(
    archive: np.lib.npyio.NpzFile, name: str
) -> NDArray[np.generic]:
    if name not in archive.files:
        raise CalibrationLoadError(
            f"NPZ calibration is missing required field: {name}"
        )
    try:
        return np.asarray(archive[name])
    except (OSError, TypeError, ValueError) as error:
        raise CalibrationLoadError(
            f"cannot safely load NPZ field {name}: {error}"
        ) from error


def _npz_optional(
    archive: np.lib.npyio.NpzFile, name: str
) -> NDArray[np.generic] | None:
    return _npz_required(archive, name) if name in archive.files else None


def _array_scalar(value: NDArray[np.generic], name: str) -> object:
    if value.size != 1:
        raise CalibrationLoadError(f"{name} must contain one scalar value")
    return value.reshape(-1)[0].item()


def _load_npz_calibration(path: Path) -> LoadedCalibration:
    try:
        with np.load(path, allow_pickle=False) as archive:
            matrix = _npz_required(archive, "camera_matrix")
            distortion = _npz_required(
                archive, "distortion_coefficients"
            )
            width = _number_as_positive_integer(
                _array_scalar(
                    _npz_required(archive, "image_width"), "image_width"
                ),
                "image_width",
            )
            height = _number_as_positive_integer(
                _array_scalar(
                    _npz_required(archive, "image_height"), "image_height"
                ),
                "image_height",
            )
            raw_corners_x = _npz_optional(archive, "internal_corners_x")
            raw_corners_y = _npz_optional(archive, "internal_corners_y")
            corners_x = (
                _number_as_positive_integer(
                    _array_scalar(raw_corners_x, "internal_corners_x"),
                    "internal_corners_x",
                )
                if raw_corners_x is not None
                else None
            )
            corners_y = (
                _number_as_positive_integer(
                    _array_scalar(raw_corners_y, "internal_corners_y"),
                    "internal_corners_y",
                )
                if raw_corners_y is not None
                else None
            )
            raw_square_size = _npz_optional(archive, "square_size_mm")
            square_size = _optional_finite_float(
                (
                    _array_scalar(raw_square_size, "square_size_mm")
                    if raw_square_size is not None
                    else None
                ),
                "square_size_mm",
                positive=True,
            )
            raw_rms = _npz_optional(archive, "opencv_rms")
            opencv_rms = _optional_finite_float(
                (
                    _array_scalar(raw_rms, "opencv_rms")
                    if raw_rms is not None
                    else None
                ),
                "OpenCV RMS",
            )
            raw_errors = _npz_optional(archive, "per_image_errors")
            mean_error: float | None = None
            if raw_errors is not None:
                errors = np.asarray(raw_errors, dtype=np.float64).reshape(-1)
                if errors.size and (
                    not np.isfinite(errors).all() or np.any(errors < 0)
                ):
                    raise CalibrationLoadError(
                        "per-image reprojection errors must be finite "
                        "and non-negative"
                    )
                if errors.size:
                    mean_error = float(np.mean(errors))
    except CalibrationLoadError:
        raise
    except (OSError, TypeError, ValueError, EOFError) as error:
        raise CalibrationLoadError(
            f"malformed NPZ calibration {path}: {error}"
        ) from error
    return LoadedCalibration(
        camera_matrix=np.asarray(matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(distortion, dtype=np.float64),
        image_width=width,
        image_height=height,
        internal_corners_x=corners_x,
        internal_corners_y=corners_y,
        square_size_mm=square_size,
        opencv_rms=opencv_rms,
        mean_reprojection_rmse_pixels=mean_error,
        source_path=path,
        source_format="npz",
    )


def _yaml_required_node(
    storage: cv2.FileStorage, name: str
) -> cv2.FileNode:
    node = storage.getNode(name)
    if node.empty():
        raise CalibrationLoadError(
            f"OpenCV YAML calibration is missing required field: {name}"
        )
    return node


def _yaml_optional_number(
    storage: cv2.FileStorage, name: str
) -> float | None:
    node = storage.getNode(name)
    return None if node.empty() else float(node.real())


def _load_yaml_calibration(path: Path) -> LoadedCalibration:
    try:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError) as error:
        raise CalibrationLoadError(
            f"malformed OpenCV YAML calibration {path}: {error}"
        ) from error
    if not storage.isOpened():
        raise CalibrationLoadError(
            f"malformed OpenCV YAML calibration: cannot open {path}"
        )
    try:
        matrix = _yaml_required_node(storage, "camera_matrix").mat()
        distortion = _yaml_required_node(
            storage, "distortion_coefficients"
        ).mat()
        width = _number_as_positive_integer(
            _yaml_required_node(storage, "image_width").real(),
            "image_width",
        )
        height = _number_as_positive_integer(
            _yaml_required_node(storage, "image_height").real(),
            "image_height",
        )
        raw_corners_x = _yaml_optional_number(
            storage, "internal_corners_x"
        )
        raw_corners_y = _yaml_optional_number(
            storage, "internal_corners_y"
        )
        corners_x = (
            _number_as_positive_integer(
                raw_corners_x, "internal_corners_x"
            )
            if raw_corners_x is not None
            else None
        )
        corners_y = (
            _number_as_positive_integer(
                raw_corners_y, "internal_corners_y"
            )
            if raw_corners_y is not None
            else None
        )
        square_size = _optional_finite_float(
            _yaml_optional_number(storage, "square_size_mm"),
            "square_size_mm",
            positive=True,
        )
        opencv_rms = _optional_finite_float(
            _yaml_optional_number(storage, "opencv_rms"), "OpenCV RMS"
        )
        mean_error = _optional_finite_float(
            _yaml_optional_number(
                storage, "mean_reprojection_rmse_pixels"
            ),
            "mean reprojection RMSE",
        )
    except CalibrationLoadError:
        raise
    except (TypeError, ValueError, cv2.error) as error:
        raise CalibrationLoadError(
            f"malformed OpenCV YAML calibration {path}: {error}"
        ) from error
    finally:
        storage.release()
    if matrix is None:
        raise CalibrationLoadError("camera matrix is missing from OpenCV YAML")
    if distortion is None:
        raise CalibrationLoadError(
            "distortion coefficients are missing from OpenCV YAML"
        )
    return LoadedCalibration(
        camera_matrix=np.asarray(matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(
            distortion, dtype=np.float64
        ),
        image_width=width,
        image_height=height,
        internal_corners_x=corners_x,
        internal_corners_y=corners_y,
        square_size_mm=square_size,
        opencv_rms=opencv_rms,
        mean_reprojection_rmse_pixels=mean_error,
        source_path=path,
        source_format="opencv-yaml",
    )


def load_calibration(calibration_path: Path) -> LoadedCalibration:
    """Load and validate Task 5 NPZ or OpenCV YAML without unsafe pickle."""
    path = calibration_path.expanduser().resolve()
    if not path.exists():
        raise CalibrationLoadError(f"calibration file does not exist: {path}")
    if not path.is_file():
        raise CalibrationLoadError(f"calibration path is not a file: {path}")
    extension = path.suffix.casefold()
    if extension == ".npz":
        return _load_npz_calibration(path)
    if extension in {".yaml", ".yml"}:
        return _load_yaml_calibration(path)
    raise CalibrationLoadError(
        f"unsupported calibration extension {path.suffix!r}; "
        "expected .npz, .yaml, or .yml"
    )


def calibration_output_paths(
    output_directory: Path, output_prefix: str
) -> CalibrationOutputPaths:
    if (
        not output_prefix
        or Path(output_prefix).name != output_prefix
        or output_prefix in {".", ".."}
    ):
        raise ValueError("output_prefix must be a non-empty filename prefix")
    directory = output_directory.expanduser().resolve()
    return CalibrationOutputPaths(
        npz=directory / f"{output_prefix}.npz",
        yaml=directory / f"{output_prefix}.yaml",
        json=directory / f"{output_prefix}.json",
    )


def calibration_json_payload(result: CalibrationResult) -> dict[str, object]:
    statistics = result.error_statistics
    return {
        "result_type": "intrinsic camera calibration only",
        "limitations": (
            "Intrinsics alone do not provide arbitrary 3D object positions "
            "or distances."
        ),
        "calibration_timestamp": result.calibration_timestamp,
        "source_manifest": result.source_manifest.as_posix(),
        "source_manifest_generated_at": result.manifest_generated_at,
        "configuration": {
            "image_width": result.resolution[0],
            "image_height": result.resolution[1],
            "internal_corners_x": result.pattern[0],
            "internal_corners_y": result.pattern[1],
            "square_size_mm": result.square_size_mm,
            "units": "millimetres",
            "calibration_flags": result.calibration_flags,
            "termination_criteria": {
                "type": result.termination_criteria[0],
                "maximum_iterations": result.termination_criteria[1],
                "epsilon": result.termination_criteria[2],
            },
        },
        "accepted_view_count": result.accepted_view_count,
        "accepted_image_names": list(result.accepted_image_names),
        "accepted_image_warnings": {
            name: list(warnings)
            for name, warnings in zip(
                result.accepted_image_names,
                result.accepted_image_warnings,
                strict=True,
            )
        },
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_coefficients": (
            result.distortion_coefficients.reshape(-1).tolist()
        ),
        "rotation_vectors": [
            vector.reshape(-1).tolist() for vector in result.rotation_vectors
        ],
        "translation_vectors": [
            vector.reshape(-1).tolist()
            for vector in result.translation_vectors
        ],
        "opencv_rms_calibration_error": result.opencv_rms,
        "reprojection_error_definition": (
            "Root mean squared Euclidean pixel distance per detected corner."
        ),
        "per_image_reprojection_rmse_pixels": [
            {
                "image": item.image_name,
                "rmse_pixels": item.rmse_pixels,
                "warnings": list(item.warnings),
            }
            for item in result.per_image_errors
        ],
        "aggregate_reprojection_rmse_pixels": {
            "mean": statistics.mean_rmse_pixels,
            "median": statistics.median_rmse_pixels,
            "minimum": statistics.minimum_rmse_pixels,
            "maximum": statistics.maximum_rmse_pixels,
            "standard_deviation": statistics.standard_deviation_pixels,
        },
        "software_versions": result.software_versions,
        "advisory_observations": list(result.advisory_observations),
    }


def _write_npz(path: Path, result: CalibrationResult) -> None:
    with path.open("wb") as output:
        np.savez_compressed(
            output,
            camera_matrix=result.camera_matrix,
            distortion_coefficients=result.distortion_coefficients,
            image_width=np.int64(result.resolution[0]),
            image_height=np.int64(result.resolution[1]),
            square_size_mm=np.float64(result.square_size_mm),
            internal_corners_x=np.int64(result.pattern[0]),
            internal_corners_y=np.int64(result.pattern[1]),
            opencv_rms=np.float64(result.opencv_rms),
            per_image_errors=np.asarray(
                [item.rmse_pixels for item in result.per_image_errors],
                dtype=np.float64,
            ),
            accepted_image_names=np.asarray(
                result.accepted_image_names, dtype=np.str_
            ),
            rotation_vectors=np.asarray(result.rotation_vectors),
            translation_vectors=np.asarray(result.translation_vectors),
        )
        output.flush()
        os.fsync(output.fileno())


def _write_yaml(path: Path, result: CalibrationResult) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise CalibrationStorageError(f"cannot open OpenCV YAML: {path}")
    try:
        storage.write("camera_matrix", result.camera_matrix)
        storage.write(
            "distortion_coefficients", result.distortion_coefficients
        )
        storage.write("image_width", result.resolution[0])
        storage.write("image_height", result.resolution[1])
        storage.write("internal_corners_x", result.pattern[0])
        storage.write("internal_corners_y", result.pattern[1])
        storage.write("square_size_mm", result.square_size_mm)
        storage.write("units", "millimetres")
        storage.write("opencv_rms", result.opencv_rms)
        storage.write(
            "mean_reprojection_rmse_pixels",
            result.error_statistics.mean_rmse_pixels,
        )
        storage.write("accepted_view_count", result.accepted_view_count)
    finally:
        storage.release()


def _write_json(path: Path, result: CalibrationResult) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(calibration_json_payload(result), output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".calibration-", suffix=suffix, dir=directory
    )
    os.close(descriptor)
    return Path(name)


def save_calibration_result(
    result: CalibrationResult,
    output_directory: Path,
    *,
    output_prefix: str = "camera_calibration",
    overwrite: bool = False,
    npz_writer: Writer = _write_npz,
    yaml_writer: Writer = _write_yaml,
    json_writer: Writer = _write_json,
) -> CalibrationOutputPaths:
    """Stage all formats, then replace targets as one rollback-capable set."""
    paths = calibration_output_paths(output_directory, output_prefix)
    targets = (paths.npz, paths.yaml, paths.json)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise CalibrationStorageError(
            "refusing to overwrite calibration outputs: "
            + ", ".join(str(path) for path in existing)
        )
    try:
        paths.npz.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CalibrationStorageError(
            f"cannot create calibration output directory: {error}"
        ) from error

    staged = {
        paths.npz: _temporary_path(paths.npz.parent, ".npz"),
        paths.yaml: _temporary_path(paths.yaml.parent, ".yaml"),
        paths.json: _temporary_path(paths.json.parent, ".json"),
    }
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        npz_writer(staged[paths.npz], result)
        yaml_writer(staged[paths.yaml], result)
        json_writer(staged[paths.json], result)
        if overwrite:
            for target in targets:
                if target.exists():
                    backup = _temporary_path(
                        target.parent, f".backup{target.suffix}"
                    )
                    backup.unlink()
                    target.replace(backup)
                    backups[target] = backup
        for target in targets:
            staged[target].replace(target)
            committed.append(target)
    except Exception as error:
        for target in committed:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                backup.replace(target)
        raise CalibrationStorageError(
            f"failed to save complete calibration output set: {error}"
        ) from error
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    return paths
