from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import inspect_dataset

from edge_ai.checkerboard import (
    CheckerboardTargetSpec,
    generate_checkerboard_image,
)

FIXED_TIME = datetime(2026, 7, 30, 4, 5, 6, tzinfo=timezone.utc)


def _write_board(path: Path, *, pixels_per_square: int = 35) -> None:
    board = generate_checkerboard_image(
        CheckerboardTargetSpec(
            pixels_per_square=pixels_per_square,
            margin_pixels=20,
        )
    )
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    board_height, board_width = board.shape
    start_x = (1280 - board_width) // 2
    start_y = (720 - board_height) // 2
    canvas[
        start_y : start_y + board_height,
        start_x : start_x + board_width,
    ] = board
    assert cv2.imwrite(str(path), canvas)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _options(
    input_directory: Path,
    annotated_directory: Path,
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> inspect_dataset.DatasetInspectionOptions:
    return inspect_dataset.DatasetInspectionOptions(
        config_path=Path("config/calibration.example.yaml"),
        input_directory=input_directory,
        annotated_directory=annotated_directory,
        manifest_path=manifest_path,
        extensions=frozenset({".png"}),
        overwrite=overwrite,
    )


def test_dataset_manifest_order_counts_warnings_and_annotations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    nested_directory = input_directory / "nested"
    nested_directory.mkdir(parents=True)
    annotated_directory = tmp_path / "annotated"
    manifest_path = tmp_path / "manifest.json"

    accepted_path = input_directory / "b_board.png"
    blank_path = input_directory / "a_blank.png"
    wrong_resolution_path = nested_directory / "c_wrong.png"
    _write_board(accepted_path)
    assert cv2.imwrite(
        str(blank_path),
        np.full((720, 1280, 3), 255, dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(wrong_resolution_path),
        np.zeros((480, 640, 3), dtype=np.uint8),
    )
    source_hashes = {
        path: _sha256(path)
        for path in (
            accepted_path,
            blank_path,
            wrong_resolution_path,
        )
    }

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dataset inspection accessed camera or GUI")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(cv2, "imshow", forbidden)
    manifest = inspect_dataset.inspect_dataset(
        _options(input_directory, annotated_directory, manifest_path),
        generated_at=FIXED_TIME,
    )
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["generated_at"] == FIXED_TIME.isoformat()
    assert manifest["total_image_count"] == 3
    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 2
    assert int(manifest["warning_count"]) >= 1
    assert manifest["accepted_images"] == ["b_board.png"]
    rejected = manifest["rejected_images"]
    assert isinstance(rejected, list)
    assert [item["image"] for item in rejected] == [
        "a_blank.png",
        "nested/c_wrong.png",
    ]
    per_image = manifest["per_image"]
    assert isinstance(per_image, list)
    assert [item["image"] for item in per_image] == [
        "a_blank.png",
        "b_board.png",
        "nested/c_wrong.png",
    ]
    assert saved_manifest["accepted_count"] == 1
    assert saved_manifest["coverage_summary"]["horizontal"] is not None
    assert len(list(annotated_directory.rglob("*.annotated.png"))) == 3
    assert all(_sha256(path) == digest for path, digest in source_hashes.items())


def test_dataset_enumeration_is_deterministic(tmp_path: Path) -> None:
    input_directory = tmp_path
    for relative_name in ("z.png", "A.png", "nested/b.png"):
        path = input_directory / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    paths = inspect_dataset.enumerate_images(
        input_directory, frozenset({".png"})
    )

    assert [
        path.relative_to(input_directory).as_posix() for path in paths
    ] == ["A.png", "nested/b.png", "z.png"]


def test_dataset_outputs_refuse_overwrite(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    _write_board(input_directory / "board.png")
    annotated_directory = tmp_path / "annotated"
    manifest_path = tmp_path / "manifest.json"
    options = _options(input_directory, annotated_directory, manifest_path)
    inspect_dataset.inspect_dataset(options, generated_at=FIXED_TIME)

    with pytest.raises(
        inspect_dataset.DatasetInspectionError,
        match="refusing to overwrite existing manifest",
    ):
        inspect_dataset.inspect_dataset(options, generated_at=FIXED_TIME)


def test_dataset_overwrite_is_explicitly_allowed(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    _write_board(input_directory / "board.png")
    annotated_directory = tmp_path / "annotated"
    manifest_path = tmp_path / "manifest.json"
    inspect_dataset.inspect_dataset(
        _options(input_directory, annotated_directory, manifest_path),
        generated_at=FIXED_TIME,
    )

    manifest = inspect_dataset.inspect_dataset(
        _options(
            input_directory,
            annotated_directory,
            manifest_path,
            overwrite=True,
        ),
        generated_at=FIXED_TIME,
    )

    assert manifest["accepted_count"] == 1
