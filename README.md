# edge-ai

`edge-ai` is a small computer-vision project targeting an NVIDIA Jetson Orin
Nano with a Logitech C920 webcam. The repository currently provides only the
Python project and development-tool bootstrap.

The Python import package is named `edge_ai` because Python package names
cannot contain hyphens.

## Windows development setup

Use a standalone CPython 3.11 installation, not Anaconda, from PowerShell.
The following command uses the default per-user location of the official
Windows installer without hard-coding a user profile:

```powershell
$python311 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
& $python311 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev-windows.txt
```

Activate the existing environment in later PowerShell sessions with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the quality checks with the repository-local tools:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src tests
```

## Jetson dependency policy

The Jetson runtime already supplies OpenCV 4.8.0, NumPy 1.26.4, and
`cv2.aruco`. Do not install a pip OpenCV package or otherwise replace the
Jetson's native OpenCV installation. `requirements-jetson.txt` is limited to
project-specific pure-Python dependencies not supplied by the Jetson.

## Task 2 configuration and software inspection

Run the supported software-only environment check without accessing a camera:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --software-only
```

To make an editable configuration, copy the example and pass the copy to the
check:

```powershell
Copy-Item config\calibration.example.yaml config\calibration.yaml
code config\calibration.yaml
.\.venv\Scripts\python.exe scripts\check_environment.py --software-only --config config\calibration.yaml
```

The physical checkerboard has 8 by 8 printed squares. OpenCV detects the
boundaries between adjacent squares, so the corresponding detection pattern is
7 by 7 internal corners.

The Task 2 environment check inspects software and configuration only.

## Task 3 camera capture

Run a Windows headless smoke test that opens the configured camera, validates a
limited number of frames, saves nothing, and displays no window:

```powershell
.\.venv\Scripts\python.exe scripts\capture_images.py --headless-smoke-test --smoke-test-frames 5
```

Run interactive Windows capture:

```powershell
.\.venv\Scripts\python.exe scripts\capture_images.py --config config\calibration.example.yaml --warmup-frames 10
```

Press `C`, `c`, or Space to save a lossless, full-resolution PNG. Press `Q`,
`q`, or Escape to exit. Existing files are not overwritten unless
`--overwrite` is explicitly supplied.

On Windows, capture tries DirectShow first, then Media Foundation, then
OpenCV's automatic backend. On Linux, it tries V4L2 and then the automatic
backend. The script reports every attempted backend and the one that
succeeded, together with requested and negotiated settings. The expected
request is MJPG at 1280 by 720 and 30 FPS. Reported FPS may differ by up to
the larger of 1 FPS or 10 percent of the request; a larger difference is
treated as a material negotiation failure.

Successful Windows webcam validation does not constitute Jetson validation.
The future Jetson syntax is expected to be the following, but remains
unverified until it runs on the Jetson:

```bash
python3 scripts/capture_images.py --headless-smoke-test --smoke-test-frames 5
python3 scripts/capture_images.py --config config/calibration.example.yaml
```

Camera calibration is still not implemented.

## Task 4 checkerboard and offline dataset inspection

Generate the configured checkerboard target:

```powershell
.\.venv\Scripts\python.exe scripts\generate_checkerboard.py --config config\calibration.example.yaml
```

The generated PNG uses 8 by 8 printed squares, which create the 7 by 7
internal corners detected by OpenCV. Print it at actual size / 100% with
fit-to-page and other printer scaling disabled. A PNG does not guarantee its
physical dimensions: measure several printed squares, then enter the measured
value as `square_size_mm` before any real calibration.

Inspect captured calibration images without modifying the originals:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py `
  --config config\calibration.example.yaml `
  --input-dir data\raw
```

Review the terminal counts, annotated previews under `output/annotated`, and
`output/dataset_manifest.json`. The manifest separates accepted images from
rejected images and records rejection reasons, refined corner coordinates,
coverage summaries, and non-fatal coverage warnings.

Undistortion is handled separately by the Task 6 validation workflow.

## Task 5 offline intrinsic calibration

Inspect calibration images with Task 4 first. Confirm that the manifest has at
least 10 accepted, diverse views and update `square_size_mm` with the measured
printed-square size before calibrating:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_camera.py `
  --config config\calibration.example.yaml `
  --manifest output\dataset_manifest.json `
  --output-dir output
```

Calibration writes three equivalent result formats:

- NPZ for NumPy/Python loading;
- OpenCV-compatible YAML for `cv2.FileStorage`;
- JSON for human review, provenance, per-image errors, and aggregate metrics.

OpenCV RMS is the optimizer's overall calibration value. The separately
reported per-image RMSE is the root mean squared Euclidean pixel distance per
checkerboard corner after reprojection. High-error views should prompt review
of blur, focus, glare, target coverage, board flatness, and pose diversity.
The workflow reports difficult views but never deletes or silently excludes
them.

Use the Task 6 validation workflow to inspect undistorted images. Intrinsic
calibration alone does not provide arbitrary 3D object positions or distances.

## Task 6 offline undistortion validation

Choose a source image captured at exactly the same resolution as the Task 5
calibration, then validate either the NPZ or OpenCV YAML result:

```powershell
.\.venv\Scripts\python.exe scripts\validate_calibration.py `
  --calibration output\camera_calibration.npz `
  --image data\raw\example.png `
  --output-dir output\validation `
  --alpha 0.0
```

The command writes a full-resolution undistorted PNG, a valid-ROI cropped PNG
when cropping is enabled and the ROI is usable, a labeled side-by-side
comparison PNG, and a JSON validation report. Pass `--no-crop` to omit the
cropped output. The source image is read only and is never modified, resized,
overwritten, or recompressed.

Alpha controls the tradeoff used by OpenCV when choosing the new camera
matrix. `0.0` generally prioritizes valid pixels and more cropping, while
`1.0` generally retains more field of view and may preserve black borders;
intermediate values trade between those outcomes.

Camera intrinsics are resolution-dependent. Task 6 therefore rejects every
resolution mismatch instead of silently resizing the image or scaling the
camera matrix. Use an image captured at the calibrated resolution, or perform
a new calibration for the required resolution.

Visually compare straight lines and object shapes near the image edges,
stretching, black borders, and the ROI crop. A visually plausible result does
not prove metric calibration accuracy. Undistortion does not establish
arbitrary 3D position or physical distance; distance measurement and YOLO
remain unimplemented.
