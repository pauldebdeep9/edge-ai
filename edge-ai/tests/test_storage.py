from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge_ai.calibration import (
    CalibrationResult,
    PerImageReprojectionError,
    ReprojectionStatistics,
)
from edge_ai.storage import (
    CalibrationStorageError,
    save_calibration_result,
)


def _result() -> CalibrationResult:
    return CalibrationResult(
        opencv_rms=0.12,
        camera_matrix=np.array(
            [[900.0, 0.0, 640.0], [0.0, 880.0, 360.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(5),
        rotation_vectors=(np.array([[0.1], [0.0], [0.0]]),),
        translation_vectors=(np.array([[0.0], [0.0], [900.0]]),),
        resolution=(1280, 720),
        pattern=(7, 7),
        square_size_mm=25.0,
        accepted_image_names=("view.png",),
        accepted_image_warnings=(("review coverage",),),
        calibration_flags=0,
        termination_criteria=(3, 100, 1e-9),
        per_image_errors=(
            PerImageReprojectionError("view.png", 0.1, ("review coverage",)),
        ),
        error_statistics=ReprojectionStatistics(0.1, 0.1, 0.1, 0.1, 0.0),
        software_versions={
            "python": "3.11.9",
            "opencv": "4.8.0",
            "numpy": "1.26.4",
        },
        source_manifest=Path("manifest.json").resolve(),
        manifest_generated_at="2026-07-30T07:00:00+00:00",
        calibration_timestamp="2026-07-30T08:00:00+00:00",
        advisory_observations=("No automatic deletion.",),
    )


def test_all_storage_formats_can_be_loaded(tmp_path: Path) -> None:
    paths = save_calibration_result(_result(), tmp_path)
    with np.load(paths.npz) as stored:
        assert stored["camera_matrix"].shape == (3, 3)
        assert stored["image_width"] == 1280
        assert stored["accepted_image_names"].tolist() == ["view.png"]

    file_storage = cv2.FileStorage(str(paths.yaml), cv2.FILE_STORAGE_READ)
    assert file_storage.isOpened()
    try:
        yaml_matrix = file_storage.getNode("camera_matrix").mat()
        yaml_width = int(file_storage.getNode("image_width").real())
    finally:
        file_storage.release()
    assert yaml_matrix.shape == (3, 3)
    assert yaml_width == 1280

    report = json.loads(paths.json.read_text(encoding="utf-8"))
    assert report["result_type"] == "intrinsic camera calibration only"
    assert report["accepted_view_count"] == 1
    assert report["camera_matrix"][0][0] == 900.0
    assert "arbitrary 3D" in report["limitations"]


def test_storage_refuses_overwrite(tmp_path: Path) -> None:
    save_calibration_result(_result(), tmp_path)

    with pytest.raises(
        CalibrationStorageError, match="refusing to overwrite"
    ):
        save_calibration_result(_result(), tmp_path)


def test_persistence_failure_leaves_no_partial_output_set(
    tmp_path: Path,
) -> None:
    def fail_yaml(_path: Path, _result: CalibrationResult) -> None:
        raise OSError("simulated YAML failure")

    with pytest.raises(
        CalibrationStorageError, match="complete calibration output set"
    ):
        save_calibration_result(_result(), tmp_path, yaml_writer=fail_yaml)

    assert not (tmp_path / "camera_calibration.npz").exists()
    assert not (tmp_path / "camera_calibration.yaml").exists()
    assert not (tmp_path / "camera_calibration.json").exists()
