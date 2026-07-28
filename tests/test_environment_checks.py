from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

import pytest
from scripts import check_environment

from edge_ai.config import PROJECT_ROOT


def _fake_module_loader(
    tmp_path: Path,
    camera_calls: list[str],
    *,
    include_charuco: bool = True,
) -> check_environment.ModuleLoader:
    cv2_module = ModuleType("cv2")
    cv2_module.__version__ = "4.8.0"
    cv2_module.__file__ = str(tmp_path / "cv2.pyd")
    aruco_attributes = {"CharucoBoard": object()} if include_charuco else {}
    setattr(cv2_module, "aruco", SimpleNamespace(**aruco_attributes))
    setattr(cv2_module, "CAP_ANY", 0)
    setattr(cv2_module, "CAP_DSHOW", 700)

    def forbidden_hardware_or_gui_call(*_args: object, **_kwargs: object) -> None:
        camera_calls.append("called")
        raise AssertionError("software-only inspection accessed hardware or a GUI")

    setattr(cv2_module, "VideoCapture", forbidden_hardware_or_gui_call)
    setattr(cv2_module, "imshow", forbidden_hardware_or_gui_call)

    numpy_module = ModuleType("numpy")
    numpy_module.__version__ = "1.26.4"
    numpy_module.__file__ = str(tmp_path / "numpy" / "__init__.py")
    modules = {"cv2": cv2_module, "numpy": numpy_module}

    def load_module(name: str) -> ModuleType:
        return modules[name]

    return load_module


def test_software_only_inspection_does_not_access_camera_or_gui(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera_calls: list[str] = []
    software_path = PROJECT_ROOT / ".pytest_cache" / "fake-software"
    module_loader = _fake_module_loader(software_path, camera_calls)
    monkeypatch.chdir(PROJECT_ROOT / ".pytest_cache")

    exit_code = check_environment.main(
        ["--software-only"], module_loader=module_loader
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert camera_calls == []
    assert "Configuration valid: True" in output
    assert "Overall status: PASS" in output


def test_missing_required_capability_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera_calls: list[str] = []
    software_path = PROJECT_ROOT / ".pytest_cache" / "fake-software"
    module_loader = _fake_module_loader(
        software_path, camera_calls, include_charuco=False
    )

    exit_code = check_environment.main(
        ["--software-only"], module_loader=module_loader
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert camera_calls == []
    assert "cv2.aruco.CharucoBoard is unavailable" in output
    assert "Overall status: FAIL" in output


def test_invalid_configuration_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera_calls: list[str] = []
    software_path = PROJECT_ROOT / ".pytest_cache" / "fake-software"
    module_loader = _fake_module_loader(software_path, camera_calls)
    with TemporaryDirectory(
        dir=PROJECT_ROOT / ".pytest_cache"
    ) as temporary_directory:
        config_path = Path(temporary_directory) / "invalid.yaml"
        config_path.write_text(
            """\
camera:
  device_index: -1
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
  raw_images: raw
  annotated_images: annotated
  output: output
""",
            encoding="utf-8",
        )

        exit_code = check_environment.main(
            ["--software-only", "--config", str(config_path)],
            module_loader=module_loader,
        )
        output = capsys.readouterr().out

    assert exit_code == 1
    assert camera_calls == []
    assert "device_index must be non-negative" in output
    assert "Overall status: FAIL" in output
