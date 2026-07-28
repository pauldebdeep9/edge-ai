"""Checkerboard target generation and offline corner detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_ai.config import CheckerboardConfig

Image = NDArray[np.generic]
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)
MIN_COVERAGE_FRACTION = 0.25


class CheckerboardError(RuntimeError):
    """Raised for target generation or offline image-processing failures."""


@dataclass(frozen=True)
class CheckerboardTargetSpec:
    """Pixel and physical dimensions for a printable checkerboard target.

    The square at the top-left of the board is black. Colours then alternate
    horizontally and vertically.
    """

    printed_squares_x: int = 8
    printed_squares_y: int = 8
    square_size_mm: float = 25.0
    pixels_per_square: int = 200
    margin_pixels: int = 100

    def __post_init__(self) -> None:
        integer_fields = (
            ("printed_squares_x", self.printed_squares_x),
            ("printed_squares_y", self.printed_squares_y),
            ("pixels_per_square", self.pixels_per_square),
            ("margin_pixels", self.margin_pixels),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.printed_squares_x < 2 or self.printed_squares_y < 2:
            raise ValueError(
                "checkerboard must contain at least 2 by 2 printed squares"
            )
        if self.pixels_per_square <= 0:
            raise ValueError("pixels_per_square must be positive")
        if self.margin_pixels < 0:
            raise ValueError("margin_pixels must be non-negative")
        if (
            isinstance(self.square_size_mm, bool)
            or not isinstance(self.square_size_mm, (int, float))
            or not isfinite(float(self.square_size_mm))
            or self.square_size_mm <= 0
        ):
            raise ValueError("square_size_mm must be a finite positive number")
        object.__setattr__(self, "square_size_mm", float(self.square_size_mm))

    @property
    def internal_corners_x(self) -> int:
        return self.printed_squares_x - 1

    @property
    def internal_corners_y(self) -> int:
        return self.printed_squares_y - 1

    @property
    def image_width_pixels(self) -> int:
        return (
            self.printed_squares_x * self.pixels_per_square
            + 2 * self.margin_pixels
        )

    @property
    def image_height_pixels(self) -> int:
        return (
            self.printed_squares_y * self.pixels_per_square
            + 2 * self.margin_pixels
        )


@dataclass(frozen=True)
class GeneratedCheckerboard:
    """Files and metadata produced for a checkerboard target."""

    image_path: Path
    metadata_path: Path
    image: NDArray[np.uint8]
    metadata: dict[str, object]


@dataclass(frozen=True)
class CornerBoundingBox:
    """Axis-aligned bounds around refined internal corners."""

    minimum_x: float
    minimum_y: float
    maximum_x: float
    maximum_y: float

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_x": self.minimum_x,
            "minimum_y": self.minimum_y,
            "maximum_x": self.maximum_x,
            "maximum_y": self.maximum_y,
        }


@dataclass(frozen=True)
class CheckerboardDetection:
    """Offline checkerboard inspection result for one image."""

    image_path: Path
    image_width: int | None
    image_height: int | None
    detection_success: bool
    corner_count: int
    refined_corners: tuple[tuple[float, float], ...]
    rejection_reason: str | None
    bounding_box: CornerBoundingBox | None
    horizontal_coverage: float | None
    vertical_coverage: float | None
    warnings: tuple[str, ...]

    def to_dict(self, *, image_identifier: str | None = None) -> dict[str, object]:
        return {
            "image": image_identifier or self.image_path.as_posix(),
            "width": self.image_width,
            "height": self.image_height,
            "accepted": self.detection_success,
            "detection_success": self.detection_success,
            "corner_count": self.corner_count,
            "refined_corners": [
                [coordinate_x, coordinate_y]
                for coordinate_x, coordinate_y in self.refined_corners
            ],
            "rejection_reason": self.rejection_reason,
            "bounding_box": (
                self.bounding_box.to_dict()
                if self.bounding_box is not None
                else None
            ),
            "coverage": {
                "horizontal": self.horizontal_coverage,
                "vertical": self.vertical_coverage,
            },
            "warnings": list(self.warnings),
        }


def target_spec_from_config(
    checkerboard: CheckerboardConfig,
    *,
    pixels_per_square: int,
    margin_pixels: int,
) -> CheckerboardTargetSpec:
    """Build a generation specification from validated project config."""
    return CheckerboardTargetSpec(
        printed_squares_x=checkerboard.printed_squares_x,
        printed_squares_y=checkerboard.printed_squares_y,
        square_size_mm=checkerboard.square_size_mm,
        pixels_per_square=pixels_per_square,
        margin_pixels=margin_pixels,
    )


def generate_checkerboard_image(
    spec: CheckerboardTargetSpec,
) -> NDArray[np.uint8]:
    """Generate an 8-bit checkerboard with a black top-left square."""
    image = np.full(
        (spec.image_height_pixels, spec.image_width_pixels),
        255,
        dtype=np.uint8,
    )
    for square_y in range(spec.printed_squares_y):
        for square_x in range(spec.printed_squares_x):
            if (square_x + square_y) % 2 != 0:
                continue
            start_x = spec.margin_pixels + square_x * spec.pixels_per_square
            start_y = spec.margin_pixels + square_y * spec.pixels_per_square
            end_x = start_x + spec.pixels_per_square
            end_y = start_y + spec.pixels_per_square
            image[start_y:end_y, start_x:end_x] = 0
    return image


def checkerboard_metadata(
    spec: CheckerboardTargetSpec,
    generated_at: datetime,
) -> dict[str, object]:
    """Build printable-target metadata without claiming printer accuracy."""
    margin_size_mm = (
        spec.margin_pixels
        / spec.pixels_per_square
        * spec.square_size_mm
    )
    return {
        "printed_squares": {
            "x": spec.printed_squares_x,
            "y": spec.printed_squares_y,
        },
        "internal_corners": {
            "x": spec.internal_corners_x,
            "y": spec.internal_corners_y,
        },
        "nominal_square_size_mm": spec.square_size_mm,
        "pixels_per_square": spec.pixels_per_square,
        "margin_pixels": spec.margin_pixels,
        "generated_pixel_dimensions": {
            "width": spec.image_width_pixels,
            "height": spec.image_height_pixels,
        },
        "intended_board_dimensions_mm": {
            "width": spec.printed_squares_x * spec.square_size_mm,
            "height": spec.printed_squares_y * spec.square_size_mm,
        },
        "intended_page_dimensions_mm": {
            "width": (
                spec.printed_squares_x * spec.square_size_mm
                + 2 * margin_size_mm
            ),
            "height": (
                spec.printed_squares_y * spec.square_size_mm
                + 2 * margin_size_mm
            ),
        },
        "starting_corner_convention": "top-left printed square is black",
        "generation_timestamp": generated_at.astimezone(
            timezone.utc
        ).isoformat(),
        "print_instruction": (
            "Print at actual size / 100%; disable fit-to-page and all "
            "automatic printer scaling."
        ),
        "measurement_instruction": (
            "A PNG does not guarantee physical millimetre dimensions. "
            "Physically measure the printed square size and enter the "
            "measured value in square_size_mm before calibration."
        ),
    }


def write_checkerboard_target(
    spec: CheckerboardTargetSpec,
    output_path: Path,
    *,
    overwrite: bool = False,
    generated_at: datetime | None = None,
) -> GeneratedCheckerboard:
    """Write a lossless PNG target and adjacent JSON metadata."""
    resolved_output = output_path.expanduser().resolve()
    if resolved_output.suffix.casefold() != ".png":
        raise CheckerboardError("checkerboard output extension must be .png")
    metadata_path = resolved_output.with_suffix(".json")
    existing_paths = [
        path for path in (resolved_output, metadata_path) if path.exists()
    ]
    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise CheckerboardError(f"refusing to overwrite existing target: {existing}")
    try:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CheckerboardError(
            f"cannot create checkerboard output directory: {error}"
        ) from error

    image = generate_checkerboard_image(spec)
    if not cv2.imwrite(str(resolved_output), image):
        raise CheckerboardError(
            f"cannot write checkerboard PNG: {resolved_output}"
        )
    timestamp = generated_at or datetime.now(timezone.utc)
    metadata = checkerboard_metadata(spec, timestamp)
    try:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise CheckerboardError(
            f"cannot write checkerboard metadata: {metadata_path}: {error}"
        ) from error
    return GeneratedCheckerboard(
        image_path=resolved_output,
        metadata_path=metadata_path,
        image=image,
        metadata=metadata,
    )


def _rejected_detection(
    image_path: Path,
    reason: str,
    *,
    width: int | None = None,
    height: int | None = None,
    corner_count: int = 0,
) -> CheckerboardDetection:
    return CheckerboardDetection(
        image_path=image_path,
        image_width=width,
        image_height=height,
        detection_success=False,
        corner_count=corner_count,
        refined_corners=(),
        rejection_reason=reason,
        bounding_box=None,
        horizontal_coverage=None,
        vertical_coverage=None,
        warnings=(),
    )


def detect_checkerboard(
    image_path: Path,
    checkerboard: CheckerboardConfig,
    *,
    expected_width: int,
    expected_height: int,
    supported_extensions: frozenset[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> CheckerboardDetection:
    """Read an image, detect corners, refine them, and assess coverage."""
    resolved_image_path = image_path.expanduser().resolve()
    if resolved_image_path.suffix.casefold() not in supported_extensions:
        return _rejected_detection(
            resolved_image_path,
            f"unsupported file extension: {resolved_image_path.suffix or '<none>'}",
        )

    image = cv2.imread(str(resolved_image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return _rejected_detection(resolved_image_path, "unreadable image")
    image_height, image_width = image.shape[:2]
    if image_width != expected_width or image_height != expected_height:
        return _rejected_detection(
            resolved_image_path,
            "incorrect resolution: "
            f"expected {expected_width}x{expected_height}, "
            f"found {image_width}x{image_height}",
            width=image_width,
            height=image_height,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (
        checkerboard.internal_corners_x,
        checkerboard.internal_corners_y,
    )
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    try:
        detected, corners = cv2.findChessboardCorners(
            gray, pattern_size, flags
        )
    except cv2.error as error:
        return _rejected_detection(
            resolved_image_path,
            f"corner detection failure: {error}",
            width=image_width,
            height=image_height,
        )
    if not detected or corners is None:
        return _rejected_detection(
            resolved_image_path,
            "corner detection failure",
            width=image_width,
            height=image_height,
        )

    expected_corner_count = (
        checkerboard.internal_corners_x * checkerboard.internal_corners_y
    )
    detected_corner_count = int(corners.shape[0])
    if detected_corner_count != expected_corner_count:
        return _rejected_detection(
            resolved_image_path,
            "incorrect detected-corner count: "
            f"expected {expected_corner_count}, found {detected_corner_count}",
            width=image_width,
            height=image_height,
            corner_count=detected_corner_count,
        )

    termination = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    try:
        refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), termination
        )
    except cv2.error as error:
        return _rejected_detection(
            resolved_image_path,
            f"corner refinement failure: {error}",
            width=image_width,
            height=image_height,
            corner_count=detected_corner_count,
        )
    flattened = np.asarray(refined, dtype=np.float64).reshape(-1, 2)
    if (
        flattened.shape != (expected_corner_count, 2)
        or not np.isfinite(flattened).all()
    ):
        return _rejected_detection(
            resolved_image_path,
            "invalid or non-finite corner coordinates",
            width=image_width,
            height=image_height,
            corner_count=int(flattened.shape[0]),
        )

    minimum_x = float(np.min(flattened[:, 0]))
    minimum_y = float(np.min(flattened[:, 1]))
    maximum_x = float(np.max(flattened[:, 0]))
    maximum_y = float(np.max(flattened[:, 1]))
    horizontal_coverage = (maximum_x - minimum_x) / image_width
    vertical_coverage = (maximum_y - minimum_y) / image_height
    warnings: list[str] = []
    if horizontal_coverage < MIN_COVERAGE_FRACTION:
        warnings.append(
            "limited horizontal checkerboard coverage; review image diversity"
        )
    if vertical_coverage < MIN_COVERAGE_FRACTION:
        warnings.append(
            "limited vertical checkerboard coverage; review image diversity"
        )

    refined_corners = tuple(
        (float(coordinate_x), float(coordinate_y))
        for coordinate_x, coordinate_y in flattened
    )
    return CheckerboardDetection(
        image_path=resolved_image_path,
        image_width=image_width,
        image_height=image_height,
        detection_success=True,
        corner_count=expected_corner_count,
        refined_corners=refined_corners,
        rejection_reason=None,
        bounding_box=CornerBoundingBox(
            minimum_x=minimum_x,
            minimum_y=minimum_y,
            maximum_x=maximum_x,
            maximum_y=maximum_y,
        ),
        horizontal_coverage=horizontal_coverage,
        vertical_coverage=vertical_coverage,
        warnings=tuple(warnings),
    )


def create_annotated_preview(
    image_path: Path,
    detection: CheckerboardDetection,
    checkerboard: CheckerboardConfig,
    *,
    fallback_width: int,
    fallback_height: int,
) -> Image:
    """Create a review image without altering or displaying the source."""
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None or source.size == 0:
        preview = np.full(
            (fallback_height, fallback_width, 3), 48, dtype=np.uint8
        )
    else:
        preview = source.copy()

    state = "ACCEPTED" if detection.detection_success else "REJECTED"
    state_colour = (0, 180, 0) if detection.detection_success else (0, 0, 220)
    if detection.detection_success:
        corners = np.asarray(
            detection.refined_corners, dtype=np.float32
        ).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(
            preview,
            (
                checkerboard.internal_corners_x,
                checkerboard.internal_corners_y,
            ),
            corners,
            True,
        )

    text_lines = [
        f"{state}: {image_path.name}",
        f"corners: {detection.corner_count}",
    ]
    if detection.rejection_reason:
        text_lines.append(detection.rejection_reason)
    text_lines.extend(detection.warnings)
    for line_index, text in enumerate(text_lines):
        position = (12, 28 + line_index * 24)
        cv2.putText(
            preview,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            state_colour,
            1,
            cv2.LINE_AA,
        )
    return cast(Image, preview)
