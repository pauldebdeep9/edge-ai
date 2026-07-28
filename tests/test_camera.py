from collections.abc import Sequence

import cv2
import numpy as np
import pytest

from edge_ai.camera import (
    BackendSpec,
    Camera,
    CameraOpenError,
    FrameReadError,
    backend_order,
    decode_fourcc,
)
from edge_ai.config import CameraConfig


class FakeCapture:
    def __init__(
        self,
        config: CameraConfig,
        *,
        opened: bool = True,
        honor_property_sets: bool = True,
        negotiated_width: int | None = None,
        negotiated_height: int | None = None,
        negotiated_fps: float | None = None,
        frames: Sequence[tuple[bool, object]] = (),
        backend_name: str = "FAKE",
    ) -> None:
        self.opened = opened
        self.honor_property_sets = honor_property_sets
        self.frames = list(frames)
        self.backend_name = backend_name
        self.set_calls: list[tuple[int, float]] = []
        self.read_calls = 0
        self.released = False
        self.properties = {
            cv2.CAP_PROP_FOURCC: float(
                cv2.VideoWriter_fourcc(*config.fourcc)
            ),
            cv2.CAP_PROP_FRAME_WIDTH: float(
                negotiated_width
                if negotiated_width is not None
                else config.width
            ),
            cv2.CAP_PROP_FRAME_HEIGHT: float(
                negotiated_height
                if negotiated_height is not None
                else config.height
            ),
            cv2.CAP_PROP_FPS: (
                negotiated_fps
                if negotiated_fps is not None
                else config.fps
            ),
        }

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        if self.honor_property_sets:
            self.properties[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self.properties.get(property_id, 0.0)

    def read(self) -> tuple[bool, object]:
        self.read_calls += 1
        if not self.frames:
            return False, None
        return self.frames.pop(0)

    def release(self) -> None:
        self.released = True

    def getBackendName(self) -> str:
        return self.backend_name


class FakeCaptureFactory:
    def __init__(self, captures: Sequence[FakeCapture]) -> None:
        self.captures = list(captures)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, index: int, backend: int) -> FakeCapture:
        self.calls.append((index, backend))
        return self.captures.pop(0)


def _small_config() -> CameraConfig:
    return CameraConfig(width=4, height=3, fps=30.0)


def _valid_frame() -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    return np.zeros((3, 4, 3), dtype=np.uint8)


def _single_backend() -> tuple[BackendSpec, ...]:
    return (BackendSpec("TEST", 999),)


def test_backend_order_on_windows() -> None:
    assert [backend.name for backend in backend_order("Windows")] == [
        "CAP_DSHOW",
        "CAP_MSMF",
        "CAP_ANY",
    ]


def test_backend_order_on_linux() -> None:
    assert [backend.name for backend in backend_order("Linux")] == [
        "CAP_V4L2",
        "CAP_ANY",
    ]


def test_falls_back_after_preferred_backend_fails() -> None:
    config = _small_config()
    preferred = FakeCapture(config, opened=False)
    fallback = FakeCapture(config, backend_name="MSMF")
    factory = FakeCaptureFactory([preferred, fallback])
    camera = Camera(
        config,
        warmup_frames=0,
        system_name="Windows",
        capture_factory=factory,
    )

    with camera as opened_camera:
        report = opened_camera.report

    assert [call[1] for call in factory.calls] == [
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
    ]
    assert [attempt.succeeded for attempt in report.backend_attempts] == [
        False,
        True,
    ]
    assert report.backend_name == "MSMF"
    assert preferred.released
    assert fallback.released


def test_failure_when_every_backend_fails() -> None:
    config = _small_config()
    captures = [FakeCapture(config, opened=False) for _ in range(3)]
    camera = Camera(
        config,
        warmup_frames=0,
        system_name="Windows",
        capture_factory=FakeCaptureFactory(captures),
    )

    with pytest.raises(CameraOpenError, match="no camera backend succeeded"):
        camera.open()

    assert all(capture.released for capture in captures)


def test_requested_properties_are_applied() -> None:
    config = _small_config()
    capture = FakeCapture(config)
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera:
        pass

    assert capture.set_calls == [
        (
            cv2.CAP_PROP_FOURCC,
            float(cv2.VideoWriter_fourcc(*config.fourcc)),
        ),
        (cv2.CAP_PROP_FRAME_WIDTH, 4.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 3.0),
        (cv2.CAP_PROP_FPS, 30.0),
    ]


def test_fourcc_decoding() -> None:
    packed_fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

    assert decode_fourcc(packed_fourcc) == "MJPG"


def test_negotiated_resolution_success() -> None:
    config = _small_config()
    capture = FakeCapture(config)
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera as opened_camera:
        report = opened_camera.report

    assert report.requested_width == 4
    assert report.requested_height == 3
    assert report.negotiated_width == 4
    assert report.negotiated_height == 3


def test_negotiated_resolution_mismatch_fails_and_releases() -> None:
    config = _small_config()
    capture = FakeCapture(
        config,
        honor_property_sets=False,
        negotiated_width=640,
        negotiated_height=480,
    )
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with pytest.raises(CameraOpenError, match="requested resolution"):
        camera.open()

    assert capture.released


def test_small_fps_variation_is_accepted() -> None:
    config = _small_config()
    capture = FakeCapture(
        config,
        honor_property_sets=False,
        negotiated_fps=29.97,
    )
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera as opened_camera:
        report = opened_camera.report

    assert report.negotiated_fps == pytest.approx(29.97)
    assert report.fps_tolerance == pytest.approx(3.0)


def test_material_fps_mismatch_fails() -> None:
    config = _small_config()
    capture = FakeCapture(
        config,
        honor_property_sets=False,
        negotiated_fps=24.0,
    )
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with pytest.raises(CameraOpenError, match="requested FPS"):
        camera.open()

    assert capture.released


@pytest.mark.parametrize(
    ("read_result", "message"),
    [
        ((False, None), "frame read failed"),
        ((True, None), "returned no frame"),
        ((True, np.empty((0, 0, 3), dtype=np.uint8)), "empty frame"),
    ],
)
def test_invalid_or_empty_frame(
    read_result: tuple[bool, object], message: str
) -> None:
    config = _small_config()
    capture = FakeCapture(config, frames=[read_result])
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera:
        with pytest.raises(FrameReadError, match=message):
            camera.read()


def test_frame_dimensions_must_match_negotiated_resolution() -> None:
    config = _small_config()
    wrong_size_frame = np.zeros((2, 4, 3), dtype=np.uint8)
    capture = FakeCapture(config, frames=[(True, wrong_size_frame)])
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera:
        with pytest.raises(FrameReadError, match="do not match negotiated"):
            camera.read()


def test_warmup_frames_are_discarded_before_capture() -> None:
    config = _small_config()
    capture = FakeCapture(
        config,
        frames=[
            (True, _valid_frame()),
            (True, _valid_frame()),
            (True, _valid_frame()),
        ],
    )
    camera = Camera(
        config,
        warmup_frames=2,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera:
        frame = camera.read()

    assert frame.shape == (3, 4, 3)
    assert capture.read_calls == 3


def test_camera_released_on_context_success() -> None:
    config = _small_config()
    capture = FakeCapture(config)
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with camera:
        assert camera.is_open

    assert capture.released
    assert not camera.is_open


def test_camera_released_after_context_exception() -> None:
    config = _small_config()
    capture = FakeCapture(config)
    camera = Camera(
        config,
        warmup_frames=0,
        backends=_single_backend(),
        capture_factory=FakeCaptureFactory([capture]),
    )

    with pytest.raises(RuntimeError, match="test error"):
        with camera:
            raise RuntimeError("test error")

    assert capture.released
    assert not camera.is_open
