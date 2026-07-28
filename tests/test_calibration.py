from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest
from scripts import calibrate_camera

from edge_ai.calibration import (
    InsufficientViewsError,
    ManifestValidationError,
    calibrate_intrinsics,
    generate_object_points,
    load_calibration_dataset,
)
from edge_ai.config import PROJECT_ROOT, ProjectConfig

FIXED_TIME = datetime(2026, 7, 30, 8, 0, 0, tzinfo=timezone.utc)


def _manifest_payload(
    *,
    view_count: int = 12,
    noise_sigma: float = 0.0,
) -> dict[str, object]:
    object_points = generate_object_points(7, 7, 25.0)
    camera_matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 880.0, 360.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    random = np.random.default_rng(12345)
    per_image: list[dict[str, object]] = []
    accepted: list[str] = []
    for index in range(view_count):
        image_name = f"view_{index:02d}.png"
        rvec = np.array(
            [0.04 * (index % 3), -0.03 * (index % 4), 0.015 * index]
        )
        tvec = np.array(
            [
                -100.0 + 20.0 * (index % 5),
                -75.0 + 18.0 * (index % 4),
                850.0 + 35.0 * index,
            ]
        )
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        points = projected.reshape(-1, 2)
        if noise_sigma:
            points = points + random.normal(0.0, noise_sigma, points.shape)
        accepted.append(image_name)
        per_image.append(
            {
                "image": image_name,
                "width": 1280,
                "height": 720,
                "accepted": True,
                "detection_success": True,
                "corner_count": 49,
                "refined_corners": points.tolist(),
                "warnings": (
                    ["limited horizontal checkerboard coverage"]
                    if index == 0
                    else []
                ),
            }
        )
    return {
        "manifest_version": 1,
        "generated_at": "2026-07-30T07:00:00+00:00",
        "expected_resolution": {"width": 1280, "height": 720},
        "expected_internal_corner_pattern": {"x": 7, "y": 7},
        "accepted_count": view_count,
        "accepted_images": list(reversed(accepted)),
        "per_image": per_image,
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_object_point_geometry() -> None:
    points = generate_object_points(7, 7, 25.0)

    assert points.shape == (49, 3)
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[1].tolist() == [25.0, 0.0, 0.0]
    assert points[7].tolist() == [0.0, 25.0, 0.0]
    assert points[-1].tolist() == [150.0, 150.0, 0.0]
    assert np.all(points[:, 2] == 0)
    assert np.isfinite(points).all()


@pytest.mark.parametrize(
    "arguments",
    [(0, 7, 25.0), (7, 0, 25.0), (7, 7, 0.0), (7, 7, float("nan"))],
)
def test_invalid_object_point_parameters(
    arguments: tuple[int, int, float],
) -> None:
    with pytest.raises(ValueError):
        generate_object_points(*arguments)


def test_valid_manifest_loading_preserves_warnings_and_sorts_names() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        manifest_path = Path(temporary_directory) / "manifest.json"
        _write_manifest(manifest_path, _manifest_payload())

        dataset = load_calibration_dataset(
            manifest_path, ProjectConfig(), minimum_views=10
        )

    assert len(dataset.observations) == 12
    assert dataset.observations[0].image_name == "view_00.png"
    assert dataset.observations[-1].image_name == "view_11.png"
    assert dataset.observations[0].warnings


def test_unsupported_manifest_version() -> None:
    payload = _manifest_payload()
    payload["manifest_version"] = 2
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "manifest.json"
        _write_manifest(path, payload)
        with pytest.raises(ManifestValidationError, match="unsupported"):
            load_calibration_dataset(path, ProjectConfig())


def test_insufficient_accepted_views() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "manifest.json"
        _write_manifest(path, _manifest_payload(view_count=3))
        with pytest.raises(InsufficientViewsError, match="insufficient"):
            load_calibration_dataset(path, ProjectConfig(), minimum_views=10)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(
                {"expected_resolution": {"width": 640, "height": 480}}
            ),
            "resolution",
        ),
        (
            lambda payload: payload.update(
                {"expected_internal_corner_pattern": {"x": 6, "y": 7}}
            ),
            "corner pattern",
        ),
        (
            lambda payload: cast_entry(payload).__setitem__(
                "refined_corners", [[1.0, 1.0]]
            ),
            "exactly 49",
        ),
        (
            lambda payload: cast_entry(payload).__setitem__(
                "refined_corners",
                [[float("nan"), 1.0]] * 49,
            ),
            "non-finite",
        ),
        (
            lambda payload: cast_per_image(payload).append(
                deepcopy(cast_per_image(payload)[0])
            ),
            "duplicate image",
        ),
    ],
)
def test_malformed_manifest_rejection(
    mutation: object, message: str
) -> None:
    payload = _manifest_payload()
    mutation(payload)  # type: ignore[operator]
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        path = Path(temporary_directory) / "manifest.json"
        _write_manifest(path, payload)
        with pytest.raises(ManifestValidationError, match=message):
            load_calibration_dataset(path, ProjectConfig())


def cast_per_image(payload: dict[str, object]) -> list[dict[str, object]]:
    value = payload["per_image"]
    assert isinstance(value, list)
    return value  # type: ignore[return-value]


def cast_entry(payload: dict[str, object]) -> dict[str, object]:
    return cast_per_image(payload)[0]


def _calibrate_from_payload(
    payload: dict[str, object],
    manifest_path: Path,
):
    _write_manifest(manifest_path, payload)
    dataset = load_calibration_dataset(
        manifest_path, ProjectConfig(), minimum_views=10
    )
    return calibrate_intrinsics(dataset, calibrated_at=FIXED_TIME)


def test_successful_synthetic_calibration_and_error_counts() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        result = _calibrate_from_payload(
            _manifest_payload(),
            Path(temporary_directory) / "manifest.json",
        )

    assert result.camera_matrix.shape == (3, 3)
    assert np.isfinite(result.camera_matrix).all()
    assert result.camera_matrix[0, 0] > 0
    assert result.camera_matrix[1, 1] > 0
    assert np.isfinite(result.distortion_coefficients).all()
    assert len(result.rotation_vectors) == 12
    assert len(result.translation_vectors) == 12
    assert len(result.per_image_errors) == 12
    assert result.error_statistics.maximum_rmse_pixels < 0.1
    assert [item.image_name for item in result.per_image_errors] == sorted(
        item.image_name for item in result.per_image_errors
    )


def test_controlled_noise_increases_reprojection_error() -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        root = Path(temporary_directory)
        clean = _calibrate_from_payload(
            _manifest_payload(), root / "clean.json"
        )
        noisy = _calibrate_from_payload(
            _manifest_payload(noise_sigma=0.3), root / "noisy.json"
        )

    assert (
        noisy.error_statistics.mean_rmse_pixels
        > clean.error_statistics.mean_rmse_pixels
    )


def test_cli_non_zero_for_invalid_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = calibrate_camera.main(
        ["--manifest", "does-not-exist.json"]
    )

    assert exit_code == 1
    assert "Intrinsic calibration failed" in capsys.readouterr().err


def test_cli_success_for_deterministic_synthetic_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        root = Path(temporary_directory)
        manifest_path = root / "manifest.json"
        _write_manifest(manifest_path, _manifest_payload())
        options = calibrate_camera.CalibrationOptions(
            config_path=Path("config/calibration.example.yaml"),
            manifest_path=manifest_path,
            output_directory=root / "results",
            output_prefix="synthetic",
            minimum_views=10,
            overwrite=False,
        )

        exit_code = calibrate_camera.run_calibration(
            options, calibrated_at=FIXED_TIME
        )

        assert (root / "results" / "synthetic.npz").exists()
        assert (root / "results" / "synthetic.yaml").exists()
        assert (root / "results" / "synthetic.json").exists()

    assert exit_code == 0
    assert "Accepted images used: 12" in capsys.readouterr().out
