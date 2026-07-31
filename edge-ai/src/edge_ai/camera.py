"""Portable, isolated OpenCV camera access."""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from edge_ai.config import CameraConfig

Frame = NDArray[np.generic]
FPS_ABSOLUTE_TOLERANCE = 1.0
FPS_RELATIVE_TOLERANCE = 0.10


class CameraError(RuntimeError):
    """Base error for camera operations."""


class CameraOpenError(CameraError):
    """Raised when no configured backend can open a valid camera."""


class CameraValidationError(CameraError):
    """Raised when negotiated camera properties are invalid."""


class FrameReadError(CameraError):
    """Raised when a camera frame cannot be read or validated."""


class CaptureDevice(Protocol):
    """The subset of cv2.VideoCapture used by the camera abstraction."""

    def isOpened(self) -> bool: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int, int], CaptureDevice]


@dataclass(frozen=True)
class BackendSpec:
    """Named OpenCV capture backend."""

    name: str
    identifier: int


@dataclass(frozen=True)
class BackendAttempt:
    """Result of one backend attempt."""

    name: str
    identifier: int
    succeeded: bool
    error: str | None = None


@dataclass(frozen=True)
class CameraReport:
    """Requested and negotiated settings for an opened camera."""

    backend_name: str
    backend_identifier: int
    camera_index: int
    requested_fourcc: str
    negotiated_fourcc: str
    requested_width: int
    requested_height: int
    negotiated_width: int
    negotiated_height: int
    requested_fps: float
    negotiated_fps: float
    fps_tolerance: float
    backend_attempts: tuple[BackendAttempt, ...]
    property_application: tuple[tuple[str, bool], ...]


def backend_order(system_name: str | None = None) -> tuple[BackendSpec, ...]:
    """Return preferred and fallback OpenCV backends for the platform."""
    normalized_system = (system_name or platform.system()).casefold()
    if normalized_system == "windows":
        return (
            BackendSpec("CAP_DSHOW", cv2.CAP_DSHOW),
            BackendSpec("CAP_MSMF", cv2.CAP_MSMF),
            BackendSpec("CAP_ANY", cv2.CAP_ANY),
        )
    if normalized_system == "linux":
        return (
            BackendSpec("CAP_V4L2", cv2.CAP_V4L2),
            BackendSpec("CAP_ANY", cv2.CAP_ANY),
        )
    return (BackendSpec("CAP_ANY", cv2.CAP_ANY),)


def decode_fourcc(value: int | float) -> str:
    """Decode OpenCV's packed FOURCC integer into four readable characters."""
    integer_value = int(round(value))
    characters = []
    for offset in range(4):
        character_code = (integer_value >> (8 * offset)) & 0xFF
        character = chr(character_code)
        characters.append(character if character.isprintable() else ".")
    return "".join(characters)


def fps_tolerance(requested_fps: float) -> float:
    """Return the material FPS mismatch threshold.

    A difference is accepted when it is no greater than the larger of
    1.0 FPS or 10 percent of the requested value.
    """
    return max(FPS_ABSOLUTE_TOLERANCE, requested_fps * FPS_RELATIVE_TOLERANCE)


def format_camera_report(report: CameraReport) -> str:
    """Format requested, negotiated, and backend-attempt camera settings."""
    attempt_lines = []
    for attempt in report.backend_attempts:
        status = "succeeded" if attempt.succeeded else "failed"
        detail = f": {attempt.error}" if attempt.error else ""
        attempt_lines.append(
            f"  {attempt.name} ({attempt.identifier}): {status}{detail}"
        )
    property_lines = [
        f"  {name}: {'accepted' if accepted else 'not acknowledged'}"
        for name, accepted in report.property_application
    ]
    return "\n".join(
        [
            "Camera backend attempts:",
            *attempt_lines,
            f"Backend succeeded: {report.backend_name} "
            f"({report.backend_identifier})",
            f"Camera index: {report.camera_index}",
            f"FOURCC requested: {report.requested_fourcc}",
            f"FOURCC negotiated: {report.negotiated_fourcc}",
            f"Resolution requested: "
            f"{report.requested_width}x{report.requested_height}",
            f"Resolution negotiated: "
            f"{report.negotiated_width}x{report.negotiated_height}",
            f"FPS requested/negotiated: "
            f"{report.requested_fps:g}/{report.negotiated_fps:g}",
            f"Material FPS tolerance: +/-{report.fps_tolerance:g}",
            "Property application:",
            *property_lines,
        ]
    )


def _default_capture_factory(index: int, backend: int) -> CaptureDevice:
    return cast(CaptureDevice, cv2.VideoCapture(index, backend))


class Camera:
    """Context-managed configured camera with validated frames."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        warmup_frames: int = 5,
        system_name: str | None = None,
        backends: Sequence[BackendSpec] | None = None,
        capture_factory: CaptureFactory = _default_capture_factory,
    ) -> None:
        if (
            isinstance(warmup_frames, bool)
            or not isinstance(warmup_frames, int)
            or warmup_frames < 0
        ):
            raise ValueError("warmup_frames must be a non-negative integer")
        self.config = config
        self.warmup_frames = warmup_frames
        self.backends = tuple(backends or backend_order(system_name))
        if not self.backends:
            raise ValueError("at least one camera backend is required")
        self._capture_factory = capture_factory
        self._capture: CaptureDevice | None = None
        self._report: CameraReport | None = None

    @property
    def report(self) -> CameraReport:
        """Return the report for the currently open camera."""
        if self._report is None:
            raise CameraError("camera is not open")
        return self._report

    @property
    def is_open(self) -> bool:
        """Return whether this instance currently owns an open capture."""
        return self._capture is not None

    def open(self) -> Camera:
        """Open the first backend that negotiates valid requested settings."""
        if self.is_open:
            raise CameraError("camera is already open")

        attempts: list[BackendAttempt] = []
        for backend in self.backends:
            capture: CaptureDevice | None = None
            try:
                capture = self._capture_factory(
                    self.config.device_index, backend.identifier
                )
                if not capture.isOpened():
                    raise CameraOpenError("camera could not be opened")

                property_application = self._apply_properties(capture)
                report = self._read_report(
                    capture,
                    backend,
                    tuple(
                        [
                            *attempts,
                            BackendAttempt(
                                backend.name, backend.identifier, succeeded=True
                            ),
                        ]
                    ),
                    property_application,
                )
                self._validate_negotiated_settings(report)
                self._capture = capture
                self._report = report
                for _ in range(self.warmup_frames):
                    self._read_validated_frame()
                return self
            except Exception as error:
                if capture is not None:
                    capture.release()
                self._capture = None
                self._report = None
                attempts.append(
                    BackendAttempt(
                        backend.name,
                        backend.identifier,
                        succeeded=False,
                        error=str(error),
                    )
                )

        details = "; ".join(
            f"{attempt.name} ({attempt.identifier}): {attempt.error}"
            for attempt in attempts
        )
        raise CameraOpenError(
            f"no camera backend succeeded for index "
            f"{self.config.device_index}: {details}"
        )

    def _apply_properties(
        self, capture: CaptureDevice
    ) -> tuple[tuple[str, bool], ...]:
        fourcc_value = cv2.VideoWriter_fourcc(  # type: ignore[attr-defined]
            *self.config.fourcc
        )
        settings = (
            ("FOURCC", cv2.CAP_PROP_FOURCC, float(fourcc_value)),
            ("width", cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width)),
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height)),
            ("FPS", cv2.CAP_PROP_FPS, self.config.fps),
        )
        return tuple(
            (name, bool(capture.set(property_id, value)))
            for name, property_id, value in settings
        )

    def _read_report(
        self,
        capture: CaptureDevice,
        backend: BackendSpec,
        attempts: tuple[BackendAttempt, ...],
        property_application: tuple[tuple[str, bool], ...],
    ) -> CameraReport:
        backend_name = backend.name
        get_backend_name = getattr(capture, "getBackendName", None)
        if callable(get_backend_name):
            try:
                reported_name = get_backend_name()
            except Exception:
                reported_name = None
            if isinstance(reported_name, str) and reported_name:
                backend_name = reported_name

        requested_fps = self.config.fps
        return CameraReport(
            backend_name=backend_name,
            backend_identifier=backend.identifier,
            camera_index=self.config.device_index,
            requested_fourcc=self.config.fourcc,
            negotiated_fourcc=decode_fourcc(
                capture.get(cv2.CAP_PROP_FOURCC)
            ),
            requested_width=self.config.width,
            requested_height=self.config.height,
            negotiated_width=int(
                round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            ),
            negotiated_height=int(
                round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            requested_fps=requested_fps,
            negotiated_fps=float(capture.get(cv2.CAP_PROP_FPS)),
            fps_tolerance=fps_tolerance(requested_fps),
            backend_attempts=attempts,
            property_application=property_application,
        )

    @staticmethod
    def _validate_negotiated_settings(report: CameraReport) -> None:
        if (
            report.negotiated_width != report.requested_width
            or report.negotiated_height != report.requested_height
        ):
            raise CameraValidationError(
                "requested resolution "
                f"{report.requested_width}x{report.requested_height} was not "
                "negotiated; camera reported "
                f"{report.negotiated_width}x{report.negotiated_height}"
            )
        fps_difference = abs(report.negotiated_fps - report.requested_fps)
        if (
            not isfinite(report.negotiated_fps)
            or report.negotiated_fps <= 0
            or fps_difference > report.fps_tolerance
        ):
            raise CameraValidationError(
                f"requested FPS {report.requested_fps:g} was not negotiated; "
                f"camera reported {report.negotiated_fps:g}, exceeding the "
                f"+/-{report.fps_tolerance:g} tolerance"
            )

    def read(self) -> Frame:
        """Read one frame and validate its contents and dimensions."""
        if self._capture is None or self._report is None:
            raise CameraError("camera is not open")
        return self._read_validated_frame()

    def _read_validated_frame(self) -> Frame:
        if self._capture is None or self._report is None:
            raise CameraError("camera is not open")
        succeeded, frame_object = self._capture.read()
        if not succeeded:
            raise FrameReadError("camera frame read failed")
        if frame_object is None:
            raise FrameReadError("camera returned no frame")
        if not isinstance(frame_object, np.ndarray):
            raise FrameReadError("camera returned an unsupported frame type")
        if frame_object.size == 0:
            raise FrameReadError("camera returned an empty frame")
        if frame_object.ndim < 2:
            raise FrameReadError("camera frame has invalid dimensions")

        actual_height = int(frame_object.shape[0])
        actual_width = int(frame_object.shape[1])
        if (
            actual_width != self._report.negotiated_width
            or actual_height != self._report.negotiated_height
        ):
            raise FrameReadError(
                "camera frame dimensions "
                f"{actual_width}x{actual_height} do not match negotiated "
                f"{self._report.negotiated_width}x"
                f"{self._report.negotiated_height}"
            )
        return cast(Frame, frame_object)

    def release(self) -> None:
        """Release the underlying camera safely and idempotently."""
        capture = self._capture
        self._capture = None
        self._report = None
        if capture is not None:
            capture.release()

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        self.release()
