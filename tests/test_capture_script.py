from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType

import numpy as np
import pytest
from scripts import capture_images

from edge_ai.camera import BackendAttempt, CameraReport, Frame
from edge_ai.config import PROJECT_ROOT, CameraConfig, ConfigError

FIXED_TIME = datetime(2026, 7, 29, 12, 34, 56, 123456, tzinfo=timezone.utc)


def _frame() -> Frame:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _camera_report() -> CameraReport:
    return CameraReport(
        backend_name="FAKE",
        backend_identifier=700,
        camera_index=0,
        requested_fourcc="MJPG",
        negotiated_fourcc="MJPG",
        requested_width=1280,
        requested_height=720,
        negotiated_width=1280,
        negotiated_height=720,
        requested_fps=30.0,
        negotiated_fps=29.97,
        fps_tolerance=3.0,
        backend_attempts=(
            BackendAttempt("CAP_DSHOW", 700, succeeded=True),
        ),
        property_application=(
            ("FOURCC", True),
            ("width", True),
            ("height", True),
            ("FPS", True),
        ),
    )


class FakeCamera:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.read_calls = 0
        self.report = _camera_report()

    def read(self) -> Frame:
        self.read_calls += 1
        return _frame()

    def __enter__(self) -> FakeCamera:
        self.entered = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited = True


class FakeCv2Api:
    def __init__(
        self,
        *,
        keys: list[int] | None = None,
        write_result: bool = True,
    ) -> None:
        self.keys = list(keys or [])
        self.write_result = write_result
        self.imshow_calls = 0
        self.wait_key_calls = 0
        self.imwrite_calls: list[tuple[str, Frame]] = []
        self.destroy_calls = 0

    def imshow(self, _window_name: str, _frame: Frame) -> None:
        self.imshow_calls += 1

    def waitKey(self, _delay: int) -> int:
        self.wait_key_calls += 1
        if not self.keys:
            raise AssertionError("interactive test ran out of keys")
        return self.keys.pop(0)

    def imwrite(self, filename: str, frame: Frame) -> bool:
        self.imwrite_calls.append((filename, frame))
        return self.write_result

    def destroyAllWindows(self) -> None:
        self.destroy_calls += 1


def _options(
    *,
    headless: bool,
    output_directory: Path | None = None,
    smoke_test_frames: int = 3,
    max_images: int | None = None,
    overwrite: bool = False,
) -> capture_images.CaptureOptions:
    return capture_images.CaptureOptions(
        config_path=Path("config/calibration.example.yaml"),
        camera_index=None,
        output_directory=output_directory,
        headless_smoke_test=headless,
        smoke_test_frames=smoke_test_frames,
        warmup_frames=2,
        max_images=max_images,
        overwrite=overwrite,
    )


def test_headless_smoke_test_never_calls_gui_or_writes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera = FakeCamera()
    fake_cv2 = FakeCv2Api()
    factory_calls: list[tuple[CameraConfig, int]] = []

    def camera_factory(
        camera_config: CameraConfig, warmup_frames: int
    ) -> FakeCamera:
        factory_calls.append((camera_config, warmup_frames))
        return camera

    exit_code = capture_images.run_capture(
        _options(headless=True),
        camera_factory=camera_factory,
        cv2_api=fake_cv2,
        now_provider=lambda: FIXED_TIME,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert camera.entered and camera.exited
    assert camera.read_calls == 3
    assert factory_calls[0][1] == 2
    assert fake_cv2.imshow_calls == 0
    assert fake_cv2.wait_key_calls == 0
    assert fake_cv2.imwrite_calls == []
    assert fake_cv2.destroy_calls == 0
    assert "Headless smoke test frames read: 3" in output


def test_interactive_capture_keys_and_cleanup() -> None:
    temporary_root = PROJECT_ROOT / ".pytest_cache"
    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        output_directory = Path(temporary_directory) / "captures"
        camera = FakeCamera()
        fake_cv2 = FakeCv2Api(
            keys=[ord("C"), ord("c"), ord(" "), ord("q")]
        )

        exit_code = capture_images.run_capture(
            _options(
                headless=False,
                output_directory=output_directory,
            ),
            camera_factory=lambda _config, _warmup: camera,
            cv2_api=fake_cv2,
            now_provider=lambda: FIXED_TIME,
        )

    assert exit_code == 0
    assert camera.read_calls == 4
    assert len(fake_cv2.imwrite_calls) == 3
    assert fake_cv2.imshow_calls == 4
    assert fake_cv2.wait_key_calls == 4
    assert fake_cv2.destroy_calls == 1
    assert camera.exited


def test_escape_exits_interactive_capture_without_saving() -> None:
    camera = FakeCamera()
    fake_cv2 = FakeCv2Api(keys=[27])

    saved = capture_images.run_interactive_capture(
        camera,
        PROJECT_ROOT / ".pytest_cache",
        max_images=None,
        overwrite=False,
        cv2_api=fake_cv2,
        now_provider=lambda: FIXED_TIME,
    )

    assert saved == 0
    assert fake_cv2.imwrite_calls == []


def test_unique_timestamped_image_names() -> None:
    first = capture_images.timestamped_image_path(
        Path("captures"), FIXED_TIME, 1
    )
    second = capture_images.timestamped_image_path(
        Path("captures"), FIXED_TIME, 2
    )

    assert first.name.endswith("_0001.png")
    assert second.name.endswith("_0002.png")
    assert first != second


def test_refuses_to_overwrite_existing_image() -> None:
    temporary_root = PROJECT_ROOT / ".pytest_cache"
    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        output_directory = Path(temporary_directory)
        existing_path = capture_images.timestamped_image_path(
            output_directory, FIXED_TIME, 1
        )
        existing_path.touch()
        fake_cv2 = FakeCv2Api()

        with pytest.raises(
            capture_images.CaptureWorkflowError,
            match="refusing to overwrite",
        ):
            capture_images.save_frame(
                _frame(),
                output_directory,
                1,
                overwrite=False,
                cv2_api=fake_cv2,
                now_provider=lambda: FIXED_TIME,
            )

    assert fake_cv2.imwrite_calls == []


def test_output_directory_is_created() -> None:
    temporary_root = PROJECT_ROOT / ".pytest_cache"
    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        output_directory = Path(temporary_directory) / "new" / "captures"

        capture_images.create_output_directory(output_directory)

        assert output_directory.is_dir()


def test_output_directory_creation_fails_for_file_path() -> None:
    temporary_root = PROJECT_ROOT / ".pytest_cache"
    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        output_path = Path(temporary_directory) / "not-a-directory"
        output_path.touch()

        with pytest.raises(
            capture_images.CaptureWorkflowError,
            match="cannot create output directory",
        ):
            capture_images.create_output_directory(output_path)


def test_failed_imwrite_raises_clear_error() -> None:
    fake_cv2 = FakeCv2Api(write_result=False)
    temporary_root = PROJECT_ROOT / ".pytest_cache"
    with TemporaryDirectory(dir=temporary_root) as temporary_directory:
        with pytest.raises(
            capture_images.CaptureWorkflowError,
            match="cannot write calibration image",
        ):
            capture_images.save_frame(
                _frame(),
                Path(temporary_directory),
                1,
                overwrite=False,
                cv2_api=fake_cv2,
                now_provider=lambda: FIXED_TIME,
            )


def test_main_returns_non_zero_for_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_capture(_options: capture_images.CaptureOptions) -> int:
        raise ConfigError("invalid test configuration")

    monkeypatch.setattr(capture_images, "run_capture", fail_capture)

    exit_code = capture_images.main(["--headless-smoke-test"])
    error_output = capsys.readouterr().err

    assert exit_code == 1
    assert "invalid test configuration" in error_output
