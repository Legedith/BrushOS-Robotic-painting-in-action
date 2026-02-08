from __future__ import annotations

from functools import partial
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_tools import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    adjust_brightness_contrast,
    adjust_gamma,
    adjust_saturation,
    auto_enhance_frame,
    apply_pipeline,
    blur_frame,
    capture_frame,
    capture_photo,
    crop_frame,
    flip_frame,
    open_camera,
    resize_frame,
    rotate_frame,
    save_image,
    sharpen_frame,
    to_grayscale,
    zoom_frame,
)


@pytest.fixture()
def result_dir(request: pytest.FixtureRequest) -> Path:
    """Create a per-test results folder."""
    path = Path("test_results") / request.node.name
    if path.exists():
        for entry in path.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def live_frame() -> np.ndarray:
    """Capture a single full HD frame from the camera."""
    capture = open_camera(
        DEFAULT_CAMERA_INDEX, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT
    )
    try:
        frame = capture_frame(capture)
    finally:
        capture.release()

    if frame.shape[:2] != (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH):
        raise RuntimeError(
            "Full HD capture failed: "
            f"got {frame.shape[1]}x{frame.shape[0]}, "
            f"expected {DEFAULT_FRAME_WIDTH}x{DEFAULT_FRAME_HEIGHT}."
        )
    return frame


@pytest.mark.live_capture
def test_live_capture_frame_full_hd(result_dir: Path) -> None:
    """Capture a raw frame at full HD resolution."""
    capture = open_camera(
        DEFAULT_CAMERA_INDEX, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT
    )
    try:
        frame = capture_frame(capture)
    finally:
        capture.release()

    assert frame.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    output_path = result_dir / "frame.png"
    saved = cv2.imwrite(str(output_path), frame)
    if not saved:
        raise RuntimeError(f"Failed to write image to {output_path}.")


@pytest.mark.live_capture
def test_live_capture_photo_full_hd(result_dir: Path) -> None:
    """Capture a photo at full HD resolution."""
    output_path = capture_photo(
        camera_index=DEFAULT_CAMERA_INDEX,
        output_dir=result_dir,
        prefix="photo_",
        suffix=".jpg",
        width=DEFAULT_FRAME_WIDTH,
        height=DEFAULT_FRAME_HEIGHT,
    )
    image = cv2.imread(str(output_path))
    if image is None:
        raise RuntimeError(f"Failed to read image at {output_path}.")

    assert image.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    (result_dir / "result.txt").write_text(str(output_path))


@pytest.mark.live_capture
def test_live_crop_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Crop a live frame and save the result."""
    crop_width = 1280
    crop_height = 720
    x0 = (DEFAULT_FRAME_WIDTH - crop_width) // 2
    y0 = (DEFAULT_FRAME_HEIGHT - crop_height) // 2
    cropped = crop_frame(live_frame, x0, y0, crop_width, crop_height)
    assert cropped.shape[:2] == (crop_height, crop_width)
    save_image(cropped, result_dir / "cropped.png")


@pytest.mark.live_capture
def test_live_zoom_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Zoom a live frame and save the result."""
    zoomed = zoom_frame(live_frame, 1.5)
    assert zoomed.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    save_image(zoomed, result_dir / "zoomed.png")


@pytest.mark.live_capture
def test_live_resize_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Resize a live frame and save the result."""
    resized = resize_frame(live_frame, 1280, 720)
    assert resized.shape[:2] == (720, 1280)
    save_image(resized, result_dir / "resized.png")


@pytest.mark.live_capture
def test_live_rotate_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Rotate a live frame and save the result."""
    rotated = rotate_frame(live_frame, 90)
    assert rotated.shape[:2] == (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
    save_image(rotated, result_dir / "rotated.png")


@pytest.mark.live_capture
def test_live_flip_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Flip a live frame and save the result."""
    flipped = flip_frame(live_frame, "horizontal")
    assert flipped.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    save_image(flipped, result_dir / "flipped.png")


@pytest.mark.live_capture
def test_live_grayscale_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Convert a live frame to grayscale and save the result."""
    grayscale = to_grayscale(live_frame)
    assert grayscale.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    save_image(grayscale, result_dir / "grayscale.png")


@pytest.mark.live_capture
def test_live_tone_adjustments_full_hd(
    live_frame: np.ndarray, result_dir: Path
) -> None:
    """Apply brightness/contrast, gamma, and saturation to a live frame."""
    bright = adjust_brightness_contrast(live_frame, brightness=20, contrast=1.1)
    gamma = adjust_gamma(live_frame, 1.8)
    saturated = adjust_saturation(live_frame, 1.2)

    save_image(bright, result_dir / "bright.png")
    save_image(gamma, result_dir / "gamma.png")
    save_image(saturated, result_dir / "saturated.png")


@pytest.mark.live_capture
def test_live_filter_adjustments_full_hd(
    live_frame: np.ndarray, result_dir: Path
) -> None:
    """Apply blur and sharpen to a live frame."""
    blurred = blur_frame(live_frame, ksize=7)
    sharpened = sharpen_frame(live_frame, amount=0.7, sigma=1.2)

    save_image(blurred, result_dir / "blurred.png")
    save_image(sharpened, result_dir / "sharpened.png")


@pytest.mark.live_capture
def test_live_pipeline_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Apply a processing pipeline to a live frame."""
    pipeline = [
        partial(crop_frame, x=320, y=180, width=1280, height=720),
        partial(resize_frame, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT),
    ]
    processed = apply_pipeline(live_frame, pipeline)
    assert processed.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    save_image(processed, result_dir / "pipeline.png")


@pytest.mark.live_capture
def test_live_auto_enhance_full_hd(live_frame: np.ndarray, result_dir: Path) -> None:
    """Auto-enhance a live frame and save before/after."""
    save_image(live_frame, result_dir / "raw.png")
    enhanced = auto_enhance_frame(live_frame)
    assert enhanced.shape[:2] == (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH)
    save_image(enhanced, result_dir / "enhanced.png")
    raw_luma = cv2.cvtColor(live_frame, cv2.COLOR_BGR2LAB)[:, :, 0]
    enhanced_luma = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)[:, :, 0]
    (result_dir / "result.txt").write_text(
        f"raw_mean_luma={raw_luma.mean():.2f}\n"
        f"enhanced_mean_luma={enhanced_luma.mean():.2f}\n"
    )
