"""Report software capabilities without accessing camera hardware."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from edge_ai.config import ConfigError, load_config  # noqa: E402

ModuleLoader = Callable[[str], ModuleType]
CAPTURE_BACKEND_CONSTANTS = (
    "CAP_ANY",
    "CAP_DSHOW",
    "CAP_MSMF",
    "CAP_V4L2",
    "CAP_GSTREAMER",
    "CAP_FFMPEG",
)


@dataclass(frozen=True)
class EnvironmentReport:
    """Software and configuration details collected without camera access."""

    operating_system: str
    architecture: str
    python_version: str
    python_executable: Path
    opencv_version: str | None
    opencv_path: Path | None
    numpy_version: str | None
    numpy_path: Path | None
    aruco_available: bool
    charuco_board_available: bool
    capture_backends: tuple[tuple[str, int], ...]
    configuration_path: Path
    configuration_valid: bool
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether every required software and config check passed."""
        return not self.errors


def _module_version(module: ModuleType) -> str | None:
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) and version else None


def _module_path(module: ModuleType) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        return None
    return Path(module_file).resolve()


def _resolve_config_argument(config_path: Path) -> Path:
    path = config_path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def inspect_software(
    config_path: Path,
    module_loader: ModuleLoader = importlib.import_module,
) -> EnvironmentReport:
    """Inspect imports and configuration without opening a camera or GUI."""
    errors: list[str] = []
    resolved_config_path = _resolve_config_argument(config_path)

    try:
        load_config(resolved_config_path)
        configuration_valid = True
    except (ConfigError, OSError, UnicodeError) as error:
        configuration_valid = False
        errors.append(f"configuration invalid: {error}")

    opencv_version: str | None = None
    opencv_path: Path | None = None
    aruco_available = False
    charuco_board_available = False
    capture_backends: list[tuple[str, int]] = []

    try:
        cv2_module = module_loader("cv2")
    except (ImportError, OSError) as error:
        errors.append(f"OpenCV import failed: {error}")
    else:
        opencv_version = _module_version(cv2_module)
        opencv_path = _module_path(cv2_module)
        if opencv_version is None:
            errors.append("OpenCV version is unavailable")
        if opencv_path is None:
            errors.append("OpenCV installation path is unavailable")

        aruco_module = getattr(cv2_module, "aruco", None)
        aruco_available = aruco_module is not None
        charuco_board_available = (
            aruco_available and hasattr(aruco_module, "CharucoBoard")
        )
        if not aruco_available:
            errors.append("required OpenCV capability cv2.aruco is unavailable")
        elif not charuco_board_available:
            errors.append(
                "required OpenCV capability cv2.aruco.CharucoBoard is unavailable"
            )

        for constant_name in CAPTURE_BACKEND_CONSTANTS:
            value = getattr(cv2_module, constant_name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                capture_backends.append((constant_name, value))

    numpy_version: str | None = None
    numpy_path: Path | None = None
    try:
        numpy_module = module_loader("numpy")
    except (ImportError, OSError) as error:
        errors.append(f"NumPy import failed: {error}")
    else:
        numpy_version = _module_version(numpy_module)
        numpy_path = _module_path(numpy_module)
        if numpy_version is None:
            errors.append("NumPy version is unavailable")
        if numpy_path is None:
            errors.append("NumPy installation path is unavailable")

    operating_system = " ".join(
        value for value in (platform.system(), platform.release()) if value
    )

    return EnvironmentReport(
        operating_system=operating_system or "unknown",
        architecture=platform.machine() or "unknown",
        python_version=platform.python_version(),
        python_executable=Path(sys.executable).resolve(),
        opencv_version=opencv_version,
        opencv_path=opencv_path,
        numpy_version=numpy_version,
        numpy_path=numpy_path,
        aruco_available=aruco_available,
        charuco_board_available=charuco_board_available,
        capture_backends=tuple(capture_backends),
        configuration_path=resolved_config_path,
        configuration_valid=configuration_valid,
        errors=tuple(errors),
    )


def print_report(report: EnvironmentReport) -> None:
    """Print an environment report suitable for local diagnostics."""
    print("Software-only environment inspection (no camera access)")
    print(f"Operating system: {report.operating_system}")
    print(f"Architecture: {report.architecture}")
    print(f"Python version: {report.python_version}")
    print(f"Python executable: {report.python_executable}")
    print(f"OpenCV version: {report.opencv_version or 'unavailable'}")
    print(f"OpenCV installation path: {report.opencv_path or 'unavailable'}")
    print(f"NumPy version: {report.numpy_version or 'unavailable'}")
    print(f"NumPy installation path: {report.numpy_path or 'unavailable'}")
    print(f"cv2.aruco available: {report.aruco_available}")
    print(
        "cv2.aruco.CharucoBoard available: "
        f"{report.charuco_board_available}"
    )
    print("OpenCV capture-backend constants:")
    if report.capture_backends:
        for name, value in report.capture_backends:
            print(f"  {name}: {value}")
    else:
        print("  unavailable")
    print(f"Configuration file: {report.configuration_path}")
    print(f"Configuration valid: {report.configuration_valid}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"  - {error}")
    print(f"Overall status: {'PASS' if report.is_valid else 'FAIL'}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Task 2 software-only command-line interface."""
    parser = argparse.ArgumentParser(
        description="Inspect edge-ai software without accessing a camera."
    )
    parser.add_argument(
        "--software-only",
        action="store_true",
        required=True,
        help="run software and configuration checks without hardware access",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration.example.yaml"),
        help=(
            "configuration file; relative paths are resolved from the "
            "repository root"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    module_loader: ModuleLoader = importlib.import_module,
) -> int:
    """Run the supported Task 2 software-only inspection."""
    arguments = build_argument_parser().parse_args(argv)
    report = inspect_software(arguments.config, module_loader=module_loader)
    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
