from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_tools import (
    adjust_brightness_contrast,
    adjust_gamma,
    adjust_saturation,
    apply_pipeline,
    auto_increment_path,
    blur_frame,
    crop_frame,
    flip_frame,
    resize_frame,
    rotate_frame,
    sharpen_frame,
    to_grayscale,
    zoom_frame,
)


@pytest.fixture()
def result_dir(request: pytest.FixtureRequest) -> Path:
    """Create a per-test results folder."""
    path = Path("test_results") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_frame(width: int = 8, height: int = 6) -> np.ndarray:
    """Create a deterministic test frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            frame[y, x] = (x * 10 % 256, y * 10 % 256, (x + y) * 5 % 256)
    return frame


def save_frame(path: Path, frame: np.ndarray) -> None:
    """Save a frame to disk."""
    saved = cv2.imwrite(str(path), frame)
    if not saved:
        raise RuntimeError(f"Failed to write image to {path}.")


def test_auto_increment_path_creates_next_index(result_dir: Path, tmp_path: Path) -> None:
    """Compute the next incremented filename."""
    prefix = "photo_"
    suffix = ".jpg"
    (tmp_path / f"{prefix}0001{suffix}").write_text("x")
    (tmp_path / f"{prefix}0003{suffix}").write_text("y")

    next_path = auto_increment_path(tmp_path, prefix, suffix)
    assert next_path.name == f"{prefix}0004{suffix}"

    (result_dir / "result.txt").write_text(str(next_path))


def test_crop_and_zoom_frame(result_dir: Path) -> None:
    """Crop and zoom frames with predictable outputs."""
    frame = make_frame()
    cropped = crop_frame(frame, 2, 1, 3, 2)
    assert cropped.shape == (2, 3, 3)
    assert (cropped[0, 0] == np.array([20, 10, 15])).all()

    zoomed = zoom_frame(frame, 2.0)
    assert zoomed.shape == frame.shape

    expected_crop = frame[1:4, 2:6]
    expected_zoom = cv2.resize(
        expected_crop, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    assert np.array_equal(zoomed, expected_zoom)

    save_frame(result_dir / "cropped.png", cropped)
    save_frame(result_dir / "zoomed.png", zoomed)


def test_rotate_and_flip_frame(result_dir: Path) -> None:
    """Rotate and flip frames consistently."""
    frame = make_frame(width=4, height=3)
    rotated = rotate_frame(frame, 90)
    assert rotated.shape == (4, 3, 3)
    assert (rotated[0, 0] == frame[2, 0]).all()

    flipped = flip_frame(frame, "horizontal")
    assert flipped.shape == frame.shape
    assert (flipped[0, 0] == frame[0, -1]).all()

    save_frame(result_dir / "rotated.png", rotated)
    save_frame(result_dir / "flipped.png", flipped)


def test_resize_and_grayscale(result_dir: Path) -> None:
    """Resize and grayscale conversions work."""
    frame = make_frame()
    resized = resize_frame(frame, 5, 4)
    assert resized.shape == (4, 5, 3)

    gray = to_grayscale(frame)
    assert gray.ndim == 2
    expected = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(gray, expected)

    save_frame(result_dir / "resized.png", resized)
    save_frame(result_dir / "grayscale.png", gray)


def test_tone_and_saturation_adjustments(result_dir: Path) -> None:
    """Adjust brightness, contrast, gamma, and saturation."""
    frame = make_frame(width=3, height=2)
    bright = adjust_brightness_contrast(frame, brightness=20, contrast=1.2)
    assert bright.shape == frame.shape
    assert bright.dtype == frame.dtype
    assert not np.array_equal(bright, frame)

    gamma_frame = np.full((1, 1, 3), 128, dtype=np.uint8)
    gamma_adjusted = adjust_gamma(gamma_frame, 2.0)
    expected_value = int(((128 / 255.0) ** (1.0 / 2.0)) * 255)
    assert abs(int(gamma_adjusted[0, 0, 0]) - expected_value) <= 1

    saturated = adjust_saturation(frame, 1.5)
    assert saturated.shape == frame.shape
    assert not np.array_equal(saturated, frame)

    save_frame(result_dir / "bright.png", bright)
    save_frame(result_dir / "gamma.png", gamma_adjusted)
    save_frame(result_dir / "saturated.png", saturated)


def test_sharpen_blur_and_pipeline(result_dir: Path) -> None:
    """Apply blur, sharpen, and pipeline transforms."""
    frame = make_frame(width=6, height=5)
    blurred = blur_frame(frame, ksize=5)
    sharpened = sharpen_frame(frame, amount=0.8, sigma=1.2)
    assert blurred.shape == frame.shape
    assert sharpened.shape == frame.shape
    assert np.sum(np.abs(blurred.astype(int) - frame.astype(int))) > 0

    pipeline = [
        partial(crop_frame, x=1, y=1, width=4, height=3),
        partial(resize_frame, width=6, height=5),
    ]
    processed = apply_pipeline(frame, pipeline)
    assert processed.shape == frame.shape

    save_frame(result_dir / "blurred.png", blurred)
    save_frame(result_dir / "sharpened.png", sharpened)
    save_frame(result_dir / "pipeline.png", processed)
