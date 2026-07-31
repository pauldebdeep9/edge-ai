"""Generate a printable checkerboard PNG and adjacent JSON metadata."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from edge_ai.checkerboard import (  # noqa: E402
    CheckerboardError,
    target_spec_from_config,
    write_checkerboard_target,
)
from edge_ai.config import ConfigError, load_config  # noqa: E402


@dataclass(frozen=True)
class GenerationOptions:
    """Validated checkerboard-generation command-line options."""

    config_path: Path
    output_path: Path
    pixels_per_square: int
    margin_pixels: int
    overwrite: bool


def _positive_integer(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return integer


def _non_negative_integer(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if integer < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return integer


def _repository_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a printable checkerboard and metadata."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/calibration.example.yaml"),
        help="configuration file, relative to the repository root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/checkerboard_target.png"),
        help="lossless PNG output, relative to the repository root",
    )
    parser.add_argument(
        "--pixels-per-square",
        type=_positive_integer,
        default=200,
        help="pixel density per printed square (default: 200)",
    )
    parser.add_argument(
        "--margin",
        type=_non_negative_integer,
        default=100,
        help="white page margin in pixels on every side (default: 100)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing PNG and metadata file",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> GenerationOptions:
    arguments = build_argument_parser().parse_args(argv)
    return GenerationOptions(
        config_path=arguments.config,
        output_path=_repository_path(arguments.output),
        pixels_per_square=arguments.pixels_per_square,
        margin_pixels=arguments.margin,
        overwrite=arguments.overwrite,
    )


def run_generation(
    options: GenerationOptions,
    *,
    generated_at: datetime | None = None,
) -> int:
    project_config = load_config(options.config_path)
    spec = target_spec_from_config(
        project_config.checkerboard,
        pixels_per_square=options.pixels_per_square,
        margin_pixels=options.margin_pixels,
    )
    generated = write_checkerboard_target(
        spec,
        options.output_path,
        overwrite=options.overwrite,
        generated_at=generated_at,
    )
    print(f"Checkerboard PNG: {generated.image_path}")
    print(f"Metadata JSON: {generated.metadata_path}")
    print(
        "Printed/internal pattern: "
        f"{spec.printed_squares_x}x{spec.printed_squares_y} squares / "
        f"{spec.internal_corners_x}x{spec.internal_corners_y} corners"
    )
    print(
        "Print at actual size / 100% with printer scaling disabled. "
        "Measure the physical printed squares and update square_size_mm "
        "before calibration."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_options(argv)
    try:
        return run_generation(options)
    except (CheckerboardError, ConfigError, OSError, ValueError) as error:
        print(f"Checkerboard generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
