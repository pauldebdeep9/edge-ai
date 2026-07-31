from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import validate_calibration

FIXED_TIME = datetime(2026, 7, 31, 9, 0, 0, tzinfo=timezone.utc)


def _write_npz(path: Path, *, width: int = 64, height: int = 48) -> None:
    np.savez_compressed(
        path,
        camera_matrix=np.array(
            [[58.0, 0.0, 31.5], [0.0, 57.0, 23.5], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.array([-0.18, 0.04, 0.0, 0.0, 0.0]),
        image_width=np.int64(width),
        image_height=np.int64(height),
        internal_corners_x=np.int64(7),
        internal_corners_y=np.int64(7),
        square_size_mm=np.float64(25.0),
        opencv_rms=np.float64(0.12),
        per_image_errors=np.array([0.1, 0.2]),
    )


def _write_image(path: Path, *, width: int = 64, height: int = 48) -> None:
    y_coordinates, x_coordinates = np.indices((height, width))
    image = np.stack(
        (
            (x_coordinates * 4) % 256,
            (y_coordinates * 5) % 256,
            ((x_coordinates + y_coordinates) * 3) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    assert cv2.imwrite(str(path), image)


def test_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        validate_calibration.build_argument_parser().parse_args(["--help"])

    assert exit_info.value.code == 0
    assert "--calibration" in capsys.readouterr().out


def test_invalid_calibration_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = validate_calibration.main(
        [
            "--calibration",
            "missing-task6-calibration.npz",
            "--image",
            "missing-task6-image.png",
        ]
    )

    assert exit_code == 1
    assert "Calibration validation failed" in capsys.readouterr().err


def test_resolution_mismatch_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.npz"
    image_path = tmp_path / "image.png"
    _write_npz(calibration_path, width=64, height=48)
    _write_image(image_path, width=32, height=24)

    exit_code = validate_calibration.main(
        [
            "--calibration",
            str(calibration_path),
            "--image",
            str(image_path),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 1
    assert "resolution-dependent" in capsys.readouterr().err


def test_deterministic_valid_cli_does_not_access_camera_or_gui(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation CLI accessed a camera or GUI")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(cv2, "imshow", forbidden)
    calibration_path = tmp_path / "calibration.npz"
    image_path = tmp_path / "image.png"
    output_directory = tmp_path / "results"
    _write_npz(calibration_path)
    _write_image(image_path)
    options = validate_calibration.ValidationOptions(
        calibration_path=calibration_path,
        image_path=image_path,
        output_directory=output_directory,
        output_prefix="cli_sample",
        alpha=0.0,
        crop=True,
        overwrite=False,
    )

    exit_code = validate_calibration.run_validation(
        options, validated_at=FIXED_TIME
    )

    assert (output_directory / "cli_sample_undistorted_full.png").exists()
    assert (
        output_directory / "cli_sample_undistorted_cropped.png"
    ).exists()
    assert (output_directory / "cli_sample_comparison.png").exists()
    assert (output_directory / "cli_sample_report.json").exists()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Calibration:" in output
    assert "Source/calibration resolution: 64x48 / 64x48" in output
    assert "Full PNG:" in output
