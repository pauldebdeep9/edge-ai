from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

import edge_ai.validation as validation
from edge_ai.config import PROJECT_ROOT
from edge_ai.storage import (
    CalibrationLoadError,
    LoadedCalibration,
    load_calibration,
)
from edge_ai.validation import (
    ImageValidationError,
    ResolutionMismatchError,
    UndistortionError,
    ValidationStorageError,
    load_validation_image,
    require_matching_resolution,
    save_validation_result,
    undistort_image,
    validate_calibration_image,
)

FIXED_TIME = datetime(2026, 7, 31, 8, 0, 0, tzinfo=timezone.utc)
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[58.0, 0.0, 31.5], [0.0, 57.0, 23.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _distortion() -> np.ndarray:
    return np.array([-0.18, 0.04, 0.001, -0.001, 0.0], dtype=np.float64)


def _npz_fields() -> dict[str, object]:
    return {
        "camera_matrix": _camera_matrix(),
        "distortion_coefficients": _distortion(),
        "image_width": np.int64(IMAGE_WIDTH),
        "image_height": np.int64(IMAGE_HEIGHT),
        "internal_corners_x": np.int64(7),
        "internal_corners_y": np.int64(7),
        "square_size_mm": np.float64(25.0),
        "opencv_rms": np.float64(0.12),
        "per_image_errors": np.array([0.1, 0.2], dtype=np.float64),
    }


def _write_npz(
    path: Path,
    *,
    updates: dict[str, object] | None = None,
    omit: frozenset[str] = frozenset(),
) -> None:
    fields = _npz_fields()
    fields.update(updates or {})
    for name in omit:
        fields.pop(name, None)
    np.savez_compressed(path, **fields)


def _write_yaml(
    path: Path,
    *,
    matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    omit: frozenset[str] = frozenset(),
) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    assert storage.isOpened()
    try:
        fields: tuple[tuple[str, object], ...] = (
            (
                "camera_matrix",
                _camera_matrix() if matrix is None else matrix,
            ),
            (
                "distortion_coefficients",
                _distortion() if distortion is None else distortion,
            ),
            ("image_width", width),
            ("image_height", height),
            ("internal_corners_x", 7),
            ("internal_corners_y", 7),
            ("square_size_mm", 25.0),
            ("opencv_rms", 0.12),
            ("mean_reprojection_rmse_pixels", 0.15),
        )
        for name, value in fields:
            if name not in omit:
                storage.write(name, value)
    finally:
        storage.release()


def _synthetic_image() -> np.ndarray:
    y_coordinates, x_coordinates = np.indices(
        (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    image = np.stack(
        (
            (x_coordinates * 4) % 256,
            (y_coordinates * 5) % 256,
            ((x_coordinates + y_coordinates) * 3) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    cv2.rectangle(image, (4, 4), (59, 43), (255, 255, 255), 1)
    cv2.line(image, (0, 0), (63, 47), (0, 0, 0), 2)
    return image


def _write_image(path: Path) -> None:
    assert cv2.imwrite(str(path), _synthetic_image())


def _loaded_calibration(
    *,
    distortion: np.ndarray | None = None,
) -> LoadedCalibration:
    return LoadedCalibration(
        camera_matrix=_camera_matrix(),
        distortion_coefficients=(
            _distortion() if distortion is None else distortion
        ),
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        internal_corners_x=7,
        internal_corners_y=7,
        square_size_mm=25.0,
        opencv_rms=0.12,
        mean_reprojection_rmse_pixels=0.15,
        source_path=Path("calibration.npz"),
        source_format="npz",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_valid_npz_without_pickle() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.npz"
        _write_npz(path)

        loaded = load_calibration(path)

    assert loaded.source_format == "npz"
    assert loaded.resolution == (IMAGE_WIDTH, IMAGE_HEIGHT)
    assert loaded.camera_matrix.shape == (3, 3)
    assert loaded.internal_corners_x == 7
    assert loaded.square_size_mm == 25.0
    assert loaded.opencv_rms == pytest.approx(0.12)
    assert loaded.mean_reprojection_rmse_pixels == pytest.approx(0.15)


def test_loads_valid_opencv_yaml() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.yaml"
        _write_yaml(path)

        loaded = load_calibration(path)

    assert loaded.source_format == "opencv-yaml"
    assert loaded.resolution == (IMAGE_WIDTH, IMAGE_HEIGHT)
    assert np.allclose(loaded.camera_matrix, _camera_matrix())
    assert np.allclose(loaded.distortion_coefficients, _distortion())
    assert loaded.mean_reprojection_rmse_pixels == pytest.approx(0.15)


def test_rejects_malformed_npz() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.npz"
        path.write_bytes(b"not an npz archive")

        with pytest.raises(CalibrationLoadError, match="malformed NPZ"):
            load_calibration(path)


def test_rejects_malformed_yaml() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.yaml"
        path.write_text("not: [valid", encoding="utf-8")

        with pytest.raises(CalibrationLoadError, match="malformed OpenCV YAML"):
            load_calibration(path)


def test_rejects_unsupported_calibration_extension() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.json"
        path.write_text("{}", encoding="utf-8")

        with pytest.raises(CalibrationLoadError, match="unsupported"):
            load_calibration(path)


def test_rejects_missing_calibration_file() -> None:
    missing = PROJECT_ROOT / ".pytest_cache" / "missing-calibration.npz"

    with pytest.raises(CalibrationLoadError, match="does not exist"):
        load_calibration(missing)


def test_rejects_invalid_camera_matrix_shape() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.npz"
        _write_npz(
            path,
            updates={"camera_matrix": np.eye(2, dtype=np.float64)},
        )

        with pytest.raises(CalibrationLoadError, match="exactly 3x3"):
            load_calibration(path)


def test_rejects_missing_distortion_coefficients() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.npz"
        _write_npz(path, omit=frozenset({"distortion_coefficients"}))

        with pytest.raises(CalibrationLoadError, match="missing required"):
            load_calibration(path)


@pytest.mark.parametrize(
    "updates",
    [
        {"camera_matrix": np.full((3, 3), np.nan)},
        {
            "distortion_coefficients": np.array(
                [0.0, np.inf, 0.0, 0.0, 0.0]
            )
        },
    ],
)
def test_rejects_non_finite_calibration_values(
    updates: dict[str, object],
) -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.npz"
        _write_npz(path, updates=updates)

        with pytest.raises(CalibrationLoadError, match="finite"):
            load_calibration(path)


@pytest.mark.parametrize(
    "updates",
    [{"image_width": np.int64(0)}, {"image_height": np.int64(-1)}],
)
def test_rejects_invalid_calibration_resolution(
    updates: dict[str, object],
) -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "calibration.npz"
        _write_npz(path, updates=updates)

        with pytest.raises(CalibrationLoadError, match="positive integer"):
            load_calibration(path)


def test_loads_readable_image_without_changing_it() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        before = _sha256(path)

        loaded = load_validation_image(path)

        assert loaded.resolution == (IMAGE_WIDTH, IMAGE_HEIGHT)
        assert loaded.pixels.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        assert _sha256(path) == before


def test_rejects_unreadable_image() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        path.write_bytes(b"not an image")

        with pytest.raises(ImageValidationError, match="unreadable"):
            load_validation_image(path)


def test_rejects_unsupported_image_extension() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.gif"
        path.write_bytes(b"GIF89a")

        with pytest.raises(ImageValidationError, match="unsupported"):
            load_validation_image(path)


def test_rejects_empty_image_array() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        path.touch()

        with pytest.raises(ImageValidationError, match="empty"):
            load_validation_image(
                path,
                image_reader=lambda _name, _flags: np.empty(
                    (0, 0, 3), dtype=np.uint8
                ),
            )


def test_exact_resolution_matches_and_mismatch_is_rejected() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        source = load_validation_image(path)
        calibration = _loaded_calibration()

        require_matching_resolution(source, calibration)
        mismatched = LoadedCalibration(
            camera_matrix=_camera_matrix(),
            distortion_coefficients=_distortion(),
            image_width=1280,
            image_height=720,
            internal_corners_x=7,
            internal_corners_y=7,
            square_size_mm=25.0,
            opencv_rms=None,
            mean_reprojection_rmse_pixels=None,
            source_path=Path("other.npz"),
            source_format="npz",
        )
        with pytest.raises(
            ResolutionMismatchError, match="resolution-dependent"
        ):
            require_matching_resolution(source, mismatched)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_undistortion_alpha_endpoints_and_dimensions(alpha: float) -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        source = load_validation_image(path)

        result = undistort_image(
            source, _loaded_calibration(), alpha=alpha, crop=True
        )

    assert result.alpha == alpha
    assert result.full_image.shape == source.pixels.shape
    assert result.full_image.size > 0
    assert np.isfinite(result.full_image).all()
    assert result.new_camera_matrix.shape == (3, 3)
    assert np.isfinite(result.new_camera_matrix).all()
    assert result.valid_roi is not None
    assert result.cropped_image is not None
    assert result.cropped_image.size > 0


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        source = load_validation_image(path)

        with pytest.raises(UndistortionError, match="inclusive range"):
            undistort_image(source, _loaded_calibration(), alpha=alpha)


def test_invalid_roi_omits_crop_but_preserves_full_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_provider = validation.cv2.getOptimalNewCameraMatrix

    def invalid_roi_provider(
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        image_size: tuple[int, int],
        alpha: float,
        new_size: tuple[int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        matrix, _roi = real_provider(
            camera_matrix,
            distortion,
            image_size,
            alpha,
            new_size,
        )
        return matrix, (0, 0, 0, 0)

    monkeypatch.setattr(
        validation.cv2,
        "getOptimalNewCameraMatrix",
        invalid_roi_provider,
    )
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        source = load_validation_image(path)

        result = undistort_image(source, _loaded_calibration(), crop=True)

    assert result.full_image.shape == source.pixels.shape
    assert result.valid_roi is None
    assert result.cropped_image is None
    assert result.warnings
    assert "invalid ROI" in result.warnings[0]


def test_no_crop_option_omits_valid_roi_crop() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        source = load_validation_image(path)

        result = undistort_image(
            source, _loaded_calibration(), crop=False
        )

    assert result.valid_roi is not None
    assert result.cropped_image is None
    assert result.crop_omission_reason == "cropping disabled by request"
    assert result.warnings == ()


def test_radial_undistortion_is_deterministic_and_changes_mapping() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "image.png"
        _write_image(path)
        source = load_validation_image(path)
        calibration = _loaded_calibration()

        first = undistort_image(source, calibration, alpha=0.25)
        second = undistort_image(source, calibration, alpha=0.25)

    assert np.array_equal(first.full_image, second.full_image)
    assert np.allclose(first.new_camera_matrix, second.new_camera_matrix)
    assert not np.array_equal(first.full_image, source.pixels)


def test_persistence_writes_reloadable_lossless_set_and_preserves_source() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "source.png"
        calibration_path = root / "calibration.npz"
        _write_image(image_path)
        _write_npz(calibration_path)
        source_digest = _sha256(image_path)

        run = validate_calibration_image(
            calibration_path,
            image_path,
            root / "results",
            output_prefix="sample",
            alpha=0.0,
            crop=True,
            validated_at=FIXED_TIME,
        )

        assert run.output_paths.cropped is not None
        full = cv2.imread(
            str(run.output_paths.full), cv2.IMREAD_UNCHANGED
        )
        cropped = cv2.imread(
            str(run.output_paths.cropped), cv2.IMREAD_UNCHANGED
        )
        comparison = cv2.imread(
            str(run.output_paths.comparison), cv2.IMREAD_UNCHANGED
        )
        report = json.loads(
            run.output_paths.report.read_text(encoding="utf-8")
        )

        assert full is not None and full.shape == (48, 64, 3)
        assert cropped is not None and cropped.size > 0
        assert comparison is not None and comparison.shape == (88, 128, 3)
        assert report["validation_timestamp"] == FIXED_TIME.isoformat()
        assert report["resolution_match"] is True
        assert report["explicit_scaling_used"] is False
        assert report["cropped_output_path"] is not None
        assert report["checkerboard_observations"] is None
        assert "Visual inspection" in report["visual_inspection_required"]
        assert "3D" in report["limitations"]
        assert _sha256(image_path) == source_digest


def test_persistence_refuses_overwrite() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "source.png"
        _write_image(image_path)
        source = load_validation_image(image_path)
        calibration = _loaded_calibration()
        result = undistort_image(source, calibration)
        save_validation_result(
            calibration,
            source,
            result,
            root / "results",
            validated_at=FIXED_TIME,
        )

        with pytest.raises(
            ValidationStorageError, match="refusing to overwrite"
        ):
            save_validation_result(
                calibration,
                source,
                result,
                root / "results",
                validated_at=FIXED_TIME,
            )


def test_persistence_failure_cleans_partial_output_set() -> None:
    calls = 0

    def fail_second_png(_path: Path, _image: np.ndarray) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated PNG write failure")
        success, encoded = cv2.imencode(".png", _image)
        assert success
        _path.write_bytes(encoded.tobytes())

    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "source.png"
        _write_image(image_path)
        source = load_validation_image(image_path)
        calibration = _loaded_calibration()
        result = undistort_image(source, calibration)
        output_directory = root / "results"

        with pytest.raises(
            ValidationStorageError, match="complete validation output set"
        ):
            save_validation_result(
                calibration,
                source,
                result,
                output_directory,
                validated_at=FIXED_TIME,
                png_writer=fail_second_png,
            )

        assert not list(output_directory.glob("*.png"))
        assert not list(output_directory.glob("*.json"))
        assert not list(output_directory.glob(".validation-*"))
