"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    integer = _require_integer(value, name)
    if integer <= 0:
        raise ConfigError(f"{name} must be positive")
    return integer


def _require_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a finite positive number")
    return number


def _require_fourcc(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError("camera.fourcc must be a non-empty string")
    if len(value) != 4 or not all(character.isascii() and character.isalnum() for character in value):
        raise ConfigError(
            "camera.fourcc must contain exactly four ASCII letters or digits"
        )
    return value


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConfigError(f"{name} must be a YAML mapping")
    return cast(Mapping[str, object], value)


def _validate_keys(
    mapping: Mapping[str, object], expected: set[str], name: str
) -> None:
    missing = expected - mapping.keys()
    unknown = mapping.keys() - expected
    if missing:
        raise ConfigError(f"{name} is missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(sorted(unknown))}")


def _resolve_path(value: object, name: str, base_directory: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


@dataclass(frozen=True)
class CameraConfig:
    """Camera settings that can be validated without opening a camera."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"

    def __post_init__(self) -> None:
        device_index = _require_integer(self.device_index, "camera.device_index")
        if device_index < 0:
            raise ConfigError("camera.device_index must be non-negative")
        _require_positive_integer(self.width, "camera.width")
        _require_positive_integer(self.height, "camera.height")
        fps = _require_positive_number(self.fps, "camera.fps")
        fourcc = _require_fourcc(self.fourcc)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "fourcc", fourcc)


@dataclass(frozen=True)
class CheckerboardConfig:
    """Printed checkerboard geometry used by later calibration tasks."""

    printed_squares_x: int = 8
    printed_squares_y: int = 8
    internal_corners_x: int = 7
    internal_corners_y: int = 7
    square_size_mm: float = 25.0

    def __post_init__(self) -> None:
        printed_x = _require_positive_integer(
            self.printed_squares_x, "checkerboard.printed_squares_x"
        )
        printed_y = _require_positive_integer(
            self.printed_squares_y, "checkerboard.printed_squares_y"
        )
        internal_x = _require_positive_integer(
            self.internal_corners_x, "checkerboard.internal_corners_x"
        )
        internal_y = _require_positive_integer(
            self.internal_corners_y, "checkerboard.internal_corners_y"
        )
        square_size = _require_positive_number(
            self.square_size_mm, "checkerboard.square_size_mm"
        )
        if internal_x != printed_x - 1:
            raise ConfigError(
                "checkerboard.internal_corners_x must equal "
                "checkerboard.printed_squares_x - 1"
            )
        if internal_y != printed_y - 1:
            raise ConfigError(
                "checkerboard.internal_corners_y must equal "
                "checkerboard.printed_squares_y - 1"
            )
        object.__setattr__(self, "square_size_mm", square_size)


@dataclass(frozen=True)
class PathsConfig:
    """Resolved project paths."""

    raw_images: Path
    annotated_images: Path
    output: Path

    @classmethod
    def default(cls) -> PathsConfig:
        """Return default paths anchored to the repository root."""
        return cls(
            raw_images=(PROJECT_ROOT / "data" / "raw").resolve(),
            annotated_images=(PROJECT_ROOT / "output" / "annotated").resolve(),
            output=(PROJECT_ROOT / "output").resolve(),
        )


@dataclass(frozen=True)
class ProjectConfig:
    """Complete validated edge-ai configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    checkerboard: CheckerboardConfig = field(default_factory=CheckerboardConfig)
    paths: PathsConfig = field(default_factory=PathsConfig.default)


def load_config(config_path: Path) -> ProjectConfig:
    """Load and validate a YAML file.

    Relative project paths in the YAML file are resolved from the directory
    containing that file, never from the process working directory.
    """
    expanded_config_path = config_path.expanduser()
    if not expanded_config_path.is_absolute():
        expanded_config_path = PROJECT_ROOT / expanded_config_path
    resolved_config_path = expanded_config_path.resolve()
    try:
        loaded: object = yaml.safe_load(
            resolved_config_path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as error:
        raise ConfigError(
            f"invalid YAML in {resolved_config_path}: {error}"
        ) from error

    root = _as_mapping(loaded, "configuration")
    _validate_keys(root, {"camera", "checkerboard", "paths"}, "configuration")

    camera = _as_mapping(root["camera"], "camera")
    _validate_keys(
        camera, {"device_index", "width", "height", "fps", "fourcc"}, "camera"
    )

    checkerboard = _as_mapping(root["checkerboard"], "checkerboard")
    _validate_keys(
        checkerboard,
        {
            "printed_squares_x",
            "printed_squares_y",
            "internal_corners_x",
            "internal_corners_y",
            "square_size_mm",
        },
        "checkerboard",
    )

    paths = _as_mapping(root["paths"], "paths")
    _validate_keys(paths, {"raw_images", "annotated_images", "output"}, "paths")
    base_directory = resolved_config_path.parent

    return ProjectConfig(
        camera=CameraConfig(
            device_index=_require_integer(
                camera["device_index"], "camera.device_index"
            ),
            width=_require_integer(camera["width"], "camera.width"),
            height=_require_integer(camera["height"], "camera.height"),
            fps=_require_positive_number(camera["fps"], "camera.fps"),
            fourcc=_require_fourcc(camera["fourcc"]),
        ),
        checkerboard=CheckerboardConfig(
            printed_squares_x=_require_integer(
                checkerboard["printed_squares_x"],
                "checkerboard.printed_squares_x",
            ),
            printed_squares_y=_require_integer(
                checkerboard["printed_squares_y"],
                "checkerboard.printed_squares_y",
            ),
            internal_corners_x=_require_integer(
                checkerboard["internal_corners_x"],
                "checkerboard.internal_corners_x",
            ),
            internal_corners_y=_require_integer(
                checkerboard["internal_corners_y"],
                "checkerboard.internal_corners_y",
            ),
            square_size_mm=_require_positive_number(
                checkerboard["square_size_mm"],
                "checkerboard.square_size_mm",
            ),
        ),
        paths=PathsConfig(
            raw_images=_resolve_path(
                paths["raw_images"], "paths.raw_images", base_directory
            ),
            annotated_images=_resolve_path(
                paths["annotated_images"],
                "paths.annotated_images",
                base_directory,
            ),
            output=_resolve_path(paths["output"], "paths.output", base_directory),
        ),
    )
