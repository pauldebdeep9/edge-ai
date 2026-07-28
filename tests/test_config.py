from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from edge_ai.config import (
    PROJECT_ROOT,
    CameraConfig,
    CheckerboardConfig,
    ConfigError,
    ProjectConfig,
    load_config,
)

EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "calibration.example.yaml"


def _yaml_with_paths(
    raw_images: str = "images/raw",
    annotated_images: str = "images/annotated",
    output: str = "results",
) -> str:
    return f"""\
camera:
  device_index: 0
  width: 1280
  height: 720
  fps: 30
  fourcc: MJPG
checkerboard:
  printed_squares_x: 8
  printed_squares_y: 8
  internal_corners_x: 7
  internal_corners_y: 7
  square_size_mm: 25.0
paths:
  raw_images: {raw_images}
  annotated_images: {annotated_images}
  output: {output}
"""


def test_valid_default_configuration() -> None:
    config = ProjectConfig()

    assert config.camera == CameraConfig()
    assert config.checkerboard == CheckerboardConfig()
    assert config.paths.raw_images == (PROJECT_ROOT / "data" / "raw").resolve()
    assert config.paths.output == (PROJECT_ROOT / "output").resolve()


def test_loads_example_yaml() -> None:
    config = load_config(EXAMPLE_CONFIG)

    assert config.camera.device_index == 0
    assert config.camera.width == 1280
    assert config.camera.height == 720
    assert config.camera.fps == 30.0
    assert config.camera.fourcc == "MJPG"
    assert config.checkerboard.internal_corners_x == 7
    assert config.checkerboard.square_size_mm == 25.0


def test_relative_config_filename_resolves_from_repository_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT / ".pytest_cache")

    config = load_config(Path("config/calibration.example.yaml"))

    assert config.camera == CameraConfig()


def test_relative_paths_resolve_from_configuration_file() -> None:
    temporary_root = PROJECT_ROOT / ".pytest_cache"
    temporary_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        config_directory = Path(temporary_directory) / "configuration"
        config_directory.mkdir()
        config_path = config_directory / "calibration.yaml"
        config_path.write_text(_yaml_with_paths(), encoding="utf-8")

        config = load_config(config_path)

        assert config.paths.raw_images == (
            config_directory / "images/raw"
        ).resolve()
        assert config.paths.annotated_images == (
            config_directory / "images/annotated"
        ).resolve()
        assert config.paths.output == (config_directory / "results").resolve()


def test_rejects_negative_camera_index() -> None:
    with pytest.raises(ConfigError, match="device_index must be non-negative"):
        CameraConfig(device_index=-1)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 720), (1280, 0), (-1, 720), (1280, -1)],
)
def test_rejects_invalid_camera_dimensions(width: int, height: int) -> None:
    with pytest.raises(ConfigError, match="must be positive"):
        CameraConfig(width=width, height=height)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ConfigError, match="finite positive"):
        CameraConfig(fps=fps)


@pytest.mark.parametrize("fourcc", ["", "ABC", "M JP", "ÅBCD"])
def test_rejects_invalid_fourcc(fourcc: str) -> None:
    with pytest.raises(ConfigError, match="camera.fourcc"):
        CameraConfig(fourcc=fourcc)


@pytest.mark.parametrize("square_size_mm", [0.0, -0.1])
def test_rejects_non_positive_square_size(square_size_mm: float) -> None:
    with pytest.raises(ConfigError, match="square_size_mm must be"):
        CheckerboardConfig(square_size_mm=square_size_mm)


@pytest.mark.parametrize(
    ("printed_squares_x", "printed_squares_y"),
    [(0, 8), (8, 0), (-1, 8), (8, -1)],
)
def test_rejects_non_positive_checkerboard_dimensions(
    printed_squares_x: int, printed_squares_y: int
) -> None:
    with pytest.raises(ConfigError, match="printed_squares_.* must be positive"):
        CheckerboardConfig(
            printed_squares_x=printed_squares_x,
            printed_squares_y=printed_squares_y,
        )


@pytest.mark.parametrize(
    ("internal_corners_x", "internal_corners_y", "axis"),
    [(6, 7, "x"), (7, 6, "y")],
)
def test_rejects_inconsistent_square_and_corner_counts(
    internal_corners_x: int, internal_corners_y: int, axis: str
) -> None:
    with pytest.raises(ConfigError, match=f"internal_corners_{axis} must equal"):
        CheckerboardConfig(
            internal_corners_x=internal_corners_x,
            internal_corners_y=internal_corners_y,
        )
