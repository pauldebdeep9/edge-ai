from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from edge_ai.checkerboard import (
    CheckerboardError,
    CheckerboardTargetSpec,
    detect_checkerboard,
    generate_checkerboard_image,
    write_checkerboard_target,
)
from edge_ai.config import CheckerboardConfig

FIXED_TIME = datetime(2026, 7, 30, 1, 2, 3, tzinfo=timezone.utc)


def _temporary_directory(tmp_path: Path) -> TemporaryDirectory[str]:
    return TemporaryDirectory(dir=tmp_path)


def _write_synthetic_board(
    image_path: Path,
    *,
    pixels_per_square: int = 60,
    perspective: bool = False,
) -> None:
    board = generate_checkerboard_image(
        CheckerboardTargetSpec(
            pixels_per_square=pixels_per_square,
            margin_pixels=20,
        )
    )
    if perspective:
        board_height, board_width = board.shape
        source_points = np.float32(
            [
                [0, 0],
                [board_width - 1, 0],
                [board_width - 1, board_height - 1],
                [0, board_height - 1],
            ]
        )
        destination_points = np.float32(
            [[320, 90], [920, 145], [850, 650], [255, 590]]
        )
        transform = cv2.getPerspectiveTransform(
            source_points, destination_points
        )
        canvas = cv2.warpPerspective(
            board,
            transform,
            (1280, 720),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
    else:
        canvas = np.full((720, 1280), 255, dtype=np.uint8)
        board_height, board_width = board.shape
        start_x = (1280 - board_width) // 2
        start_y = (720 - board_height) // 2
        canvas[
            start_y : start_y + board_height,
            start_x : start_x + board_width,
        ] = board
    assert cv2.imwrite(str(image_path), canvas)


def test_generated_board_has_exact_square_and_corner_contract() -> None:
    spec = CheckerboardTargetSpec(
        printed_squares_x=8,
        printed_squares_y=8,
        pixels_per_square=20,
        margin_pixels=10,
    )
    image = generate_checkerboard_image(spec)

    assert spec.internal_corners_x == 7
    assert spec.internal_corners_y == 7
    assert image.shape == (180, 180)


def test_generated_board_alternates_from_black_top_left() -> None:
    spec = CheckerboardTargetSpec(
        pixels_per_square=10,
        margin_pixels=5,
    )
    image = generate_checkerboard_image(spec)

    assert image[10, 10] == 0
    assert image[10, 20] == 255
    assert image[20, 10] == 255
    assert image[20, 20] == 0
    assert image[0, 0] == 255


@pytest.mark.parametrize(
    "arguments",
    [
        {"printed_squares_x": 1},
        {"printed_squares_y": 1},
        {"pixels_per_square": 0},
        {"margin_pixels": -1},
        {"square_size_mm": 0.0},
        {"square_size_mm": float("nan")},
    ],
)
def test_invalid_generation_parameters(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CheckerboardTargetSpec(**arguments)  # type: ignore[arg-type]


def test_target_write_refuses_overwrite_and_writes_metadata(
    tmp_path: Path,
) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        output_path = Path(temporary_directory) / "target.png"
        spec = CheckerboardTargetSpec(
            pixels_per_square=20,
            margin_pixels=10,
        )

        generated = write_checkerboard_target(
            spec, output_path, generated_at=FIXED_TIME
        )
        metadata = json.loads(
            generated.metadata_path.read_text(encoding="utf-8")
        )

        assert generated.image_path.exists()
        assert metadata["printed_squares"] == {"x": 8, "y": 8}
        assert metadata["internal_corners"] == {"x": 7, "y": 7}
        assert metadata["nominal_square_size_mm"] == 25.0
        assert metadata["generated_pixel_dimensions"] == {
            "width": 180,
            "height": 180,
        }
        assert metadata["intended_board_dimensions_mm"] == {
            "width": 200.0,
            "height": 200.0,
        }
        assert metadata["generation_timestamp"] == FIXED_TIME.isoformat()
        assert "actual size / 100%" in metadata["print_instruction"]
        assert "Physically measure" in metadata["measurement_instruction"]

        with pytest.raises(CheckerboardError, match="refusing to overwrite"):
            write_checkerboard_target(
                spec, output_path, generated_at=FIXED_TIME
            )


def test_target_write_requires_png_extension(tmp_path: Path) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        with pytest.raises(CheckerboardError, match="extension must be .png"):
            write_checkerboard_target(
                CheckerboardTargetSpec(),
                Path(temporary_directory) / "target.jpg",
                generated_at=FIXED_TIME,
            )


def test_detects_front_facing_board_with_refined_finite_corners(
    tmp_path: Path,
) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "front.png"
        _write_synthetic_board(image_path)

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    corners = np.asarray(result.refined_corners)
    assert result.detection_success
    assert result.corner_count == 49
    assert corners.shape == (49, 2)
    assert np.isfinite(corners).all()
    assert result.bounding_box is not None


def test_detects_controlled_perspective_board(tmp_path: Path) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "perspective.png"
        _write_synthetic_board(image_path, perspective=True)

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    assert result.detection_success
    assert result.corner_count == 49


def test_limited_coverage_is_a_warning_not_rejection(tmp_path: Path) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "small.png"
        _write_synthetic_board(image_path, pixels_per_square=35)

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    assert result.detection_success
    assert result.warnings
    assert any("limited horizontal" in warning for warning in result.warnings)


def test_detection_failure_on_blank_image(tmp_path: Path) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "blank.png"
        blank = np.full((720, 1280, 3), 255, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), blank)

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    assert not result.detection_success
    assert result.rejection_reason == "corner detection failure"


def test_unreadable_image_handling(tmp_path: Path) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "missing.png"

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    assert not result.detection_success
    assert result.rejection_reason == "unreadable image"


def test_incorrect_resolution_is_rejected(tmp_path: Path) -> None:
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "small.png"
        assert cv2.imwrite(
            str(image_path), np.zeros((480, 640, 3), dtype=np.uint8)
        )

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    assert not result.detection_success
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.rejection_reason is not None
    assert "incorrect resolution" in result.rejection_reason


def test_offline_checkerboard_functions_do_not_access_camera_or_gui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline checkerboard code accessed camera or GUI")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(cv2, "imshow", forbidden)
    with _temporary_directory(tmp_path) as temporary_directory:
        image_path = Path(temporary_directory) / "front.png"
        _write_synthetic_board(image_path)

        result = detect_checkerboard(
            image_path,
            CheckerboardConfig(),
            expected_width=1280,
            expected_height=720,
        )

    assert result.detection_success
