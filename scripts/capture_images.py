"""Capture unmodified calibration images through the isolated camera API."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

import cv2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from edge_ai.camera import (  # noqa: E402
    Camera,
    CameraError,
    CameraReport,
    Frame,
    format_camera_report,
)
from edge_ai.config import CameraConfig, ConfigError, load_config  # noqa: E402


class CaptureWorkflowError(RuntimeError):
    """Raised for capture workflow or image-writing failures."""


class CameraContext(Protocol):
    """Context-managed camera operations used by the capture workflow."""

    @property
    def report(self) -> CameraReport: ...

    def read(self) -> Frame: ...

    def __enter__(self) -> CameraContext: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class CaptureCv2Api(Protocol):
    """OpenCV GUI and file functions used only by the capture script."""

    def imshow(self, window_name: str, frame: Frame) -> None: ...

    def waitKey(self, delay: int) -> int: ...

    def imwrite(self, filename: str, frame: Frame) -> bool: ...

    def destroyAllWindows(self) -> None: ...


CameraFactory = Callable[[CameraConfig, int], CameraContext]
NowProvider = Callable[[], datetime]


@dataclass(frozen=True)
class CaptureOptions:
    """Validated command-line capture options."""

    config_path: Path
    camera_index: int | None
    output_directory: Path | None
    headless_smoke_test: bool
    smoke_test_frames: int
    warmup_frames: int
    max_images: int | None
    overwrite: bool


def _non_negative_integer(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if integer < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return integer


def _positive_integer(value: str) -> int:
    integer = _non_negative_integer(value)
    if integer == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return integer


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Task 3 camera capture command-line interface."""
    parser = argparse.ArgumentParser(
        description="Capture full-resolution calibration images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration.example.yaml"),
        help="configuration file, relative to the repository root",
    )
    parser.add_argument(
        "--camera-index",
        type=_non_negative_integer,
        help="override the configured camera device index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the configured raw-images output directory",
    )
    parser.add_argument(
        "--headless-smoke-test",
        action="store_true",
        help="read a limited number of frames without GUI or image writes",
    )
    parser.add_argument(
        "--smoke-test-frames",
        type=_positive_integer,
        default=5,
        help="number of frames to validate in headless mode (default: 5)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=_non_negative_integer,
        default=10,
        help="number of frames discarded before capture (default: 10)",
    )
    parser.add_argument(
        "--max-images",
        type=_positive_integer,
        help="stop interactive capture after this many saved images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement only if a generated filename already exists",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> CaptureOptions:
    """Parse validated command-line arguments."""
    arguments = build_argument_parser().parse_args(argv)
    return CaptureOptions(
        config_path=arguments.config,
        camera_index=arguments.camera_index,
        output_directory=arguments.output_dir,
        headless_smoke_test=arguments.headless_smoke_test,
        smoke_test_frames=arguments.smoke_test_frames,
        warmup_frames=arguments.warmup_frames,
        max_images=arguments.max_images,
        overwrite=arguments.overwrite,
    )


def _default_camera_factory(
    camera_config: CameraConfig, warmup_frames: int
) -> CameraContext:
    return Camera(camera_config, warmup_frames=warmup_frames)


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_output_directory(
    configured_directory: Path, override: Path | None
) -> Path:
    """Resolve an optional output override independently of the shell CWD."""
    if override is None:
        return configured_directory.resolve()
    output_directory = override.expanduser()
    if not output_directory.is_absolute():
        output_directory = REPOSITORY_ROOT / output_directory
    return output_directory.resolve()


def create_output_directory(output_directory: Path) -> None:
    """Create the output directory or raise a user-facing workflow error."""
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CaptureWorkflowError(
            f"cannot create output directory {output_directory}: {error}"
        ) from error
    if not output_directory.is_dir():
        raise CaptureWorkflowError(
            f"output path is not a directory: {output_directory}"
        )


def timestamped_image_path(
    output_directory: Path,
    captured_at: datetime,
    sequence_number: int,
) -> Path:
    """Build a timestamped PNG path unique within a capture session."""
    timestamp = captured_at.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    return output_directory / (
        f"calibration_{timestamp}_{sequence_number:04d}.png"
    )


def save_frame(
    frame: Frame,
    output_directory: Path,
    sequence_number: int,
    *,
    overwrite: bool,
    cv2_api: CaptureCv2Api,
    now_provider: NowProvider,
) -> Path:
    """Save an unmodified frame as PNG without silent replacement."""
    image_path = timestamped_image_path(
        output_directory, now_provider(), sequence_number
    )
    if image_path.exists() and not overwrite:
        raise CaptureWorkflowError(
            f"refusing to overwrite existing image: {image_path}"
        )
    try:
        written = cv2_api.imwrite(str(image_path), frame)
    except Exception as error:
        raise CaptureWorkflowError(
            f"cannot write calibration image {image_path}: {error}"
        ) from error
    if not written:
        raise CaptureWorkflowError(
            f"cannot write calibration image {image_path}"
        )
    return image_path


def run_headless_smoke_test(
    camera: CameraContext, frame_count: int
) -> None:
    """Read and validate frames without invoking GUI or image functions."""
    for _ in range(frame_count):
        camera.read()
    print(f"Headless smoke test frames read: {frame_count}")


def run_interactive_capture(
    camera: CameraContext,
    output_directory: Path,
    *,
    max_images: int | None,
    overwrite: bool,
    cv2_api: CaptureCv2Api,
    now_provider: NowProvider,
) -> int:
    """Display full frames and save PNGs for capture key presses."""
    saved_images = 0
    while True:
        frame = camera.read()
        try:
            cv2_api.imshow("edge-ai calibration capture", frame)
            key = cv2_api.waitKey(1) & 0xFF
        except Exception as error:
            raise CaptureWorkflowError(
                f"interactive camera display failed: {error}"
            ) from error

        if key in (ord("c"), ord("C"), ord(" ")):
            next_sequence_number = saved_images + 1
            image_path = save_frame(
                frame,
                output_directory,
                next_sequence_number,
                overwrite=overwrite,
                cv2_api=cv2_api,
                now_provider=now_provider,
            )
            saved_images = next_sequence_number
            print(f"Saved image {saved_images}: {image_path}")
            if max_images is not None and saved_images >= max_images:
                break
        elif key in (ord("q"), ord("Q"), 27):
            break
    print(f"Images saved: {saved_images}")
    return saved_images


def run_capture(
    options: CaptureOptions,
    *,
    camera_factory: CameraFactory = _default_camera_factory,
    cv2_api: CaptureCv2Api = cast(CaptureCv2Api, cv2),
    now_provider: NowProvider = _default_now,
) -> int:
    """Run headless or interactive capture with guaranteed cleanup."""
    project_config = load_config(options.config_path)
    camera_config = project_config.camera
    if options.camera_index is not None:
        camera_config = replace(
            camera_config, device_index=options.camera_index
        )

    output_directory = resolve_output_directory(
        project_config.paths.raw_images, options.output_directory
    )
    if not options.headless_smoke_test:
        create_output_directory(output_directory)

    try:
        camera_context = camera_factory(
            camera_config, options.warmup_frames
        )
        with camera_context as camera:
            print(format_camera_report(camera.report))
            if options.headless_smoke_test:
                run_headless_smoke_test(
                    camera, options.smoke_test_frames
                )
            else:
                run_interactive_capture(
                    camera,
                    output_directory,
                    max_images=options.max_images,
                    overwrite=options.overwrite,
                    cv2_api=cv2_api,
                    now_provider=now_provider,
                )
    finally:
        if not options.headless_smoke_test:
            try:
                cv2_api.destroyAllWindows()
            except Exception as error:
                raise CaptureWorkflowError(
                    f"cannot close OpenCV windows: {error}"
                ) from error
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the capture script and convert expected failures to exit code 1."""
    options = parse_options(argv)
    try:
        return run_capture(options)
    except (CameraError, CaptureWorkflowError, ConfigError, OSError) as error:
        print(f"Capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
