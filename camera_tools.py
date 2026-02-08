from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

Frame = np.ndarray
FrameProcessor = Callable[[Frame], Frame]

DEFAULT_CAMERA_INDEX = 2
CAPTURE_ROOT = Path("captures")
PHOTO_DIR = CAPTURE_ROOT / "photos"
VIDEO_DIR = CAPTURE_ROOT / "videos"


def auto_increment_path(directory: Path, prefix: str, suffix: str, digits: int = 4) -> Path:
    """Return the next available path with an incremented counter."""
    if digits < 1:
        raise ValueError("digits must be >= 1.")

    directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}}){re.escape(suffix)}$")
    max_value = 0
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if match:
            max_value = max(max_value, int(match.group(1)))

    next_value = max_value + 1
    return directory / f"{prefix}{next_value:0{digits}d}{suffix}"


def _set_capture_property(
    capture: cv2.VideoCapture,
    prop_id: int,
    value: float,
    name: str,
    tolerance: float,
) -> None:
    """Set and validate a camera property."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    if not capture.set(prop_id, float(value)):
        raise RuntimeError(f"Failed to set camera {name}.")
    actual = capture.get(prop_id)
    if actual <= 0:
        raise RuntimeError(f"Camera {name} is unavailable.")
    if abs(actual - float(value)) > tolerance:
        raise RuntimeError(f"Camera {name} mismatch: requested {value}, got {actual:.2f}.")


def open_camera(
    camera_index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    backend: int | None = None,
) -> cv2.VideoCapture:
    """Open a camera and apply optional capture settings."""
    capture = (
        cv2.VideoCapture(camera_index, backend)
        if backend is not None
        else cv2.VideoCapture(camera_index)
    )
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Camera {camera_index} could not be opened.")

    try:
        if width is not None:
            _set_capture_property(capture, cv2.CAP_PROP_FRAME_WIDTH, width, "width", 1.0)
        if height is not None:
            _set_capture_property(capture, cv2.CAP_PROP_FRAME_HEIGHT, height, "height", 1.0)
        if fps is not None:
            _set_capture_property(capture, cv2.CAP_PROP_FPS, fps, "fps", 0.5)
    except Exception:
        capture.release()
        raise

    return capture


def get_capture_info(capture: cv2.VideoCapture) -> dict[str, float]:
    """Return basic capture properties."""
    return {
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": capture.get(cv2.CAP_PROP_FPS),
    }


def capture_frame(capture: cv2.VideoCapture) -> Frame:
    """Read a single frame from an open camera."""
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def save_image(frame: Frame, output_path: Path) -> None:
    """Save an image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), frame)
    if not saved:
        raise RuntimeError(f"Failed to write image to {output_path}.")


def apply_pipeline(frame: Frame, steps: Sequence[FrameProcessor]) -> Frame:
    """Apply a sequence of frame processors."""
    output = frame
    for step in steps:
        output = step(output)
    return output


def crop_frame(frame: Frame, x: int, y: int, width: int, height: int) -> Frame:
    """Crop a frame to the specified rectangle."""
    if width <= 0 or height <= 0:
        raise ValueError("Crop width and height must be positive.")
    if x < 0 or y < 0:
        raise ValueError("Crop origin must be non-negative.")

    frame_height, frame_width = frame.shape[:2]
    if x + width > frame_width or y + height > frame_height:
        raise ValueError("Crop rectangle is outside the frame bounds.")

    return frame[y : y + height, x : x + width]


def zoom_frame(frame: Frame, factor: float) -> Frame:
    """Zoom into the center of a frame."""
    if factor < 1.0:
        raise ValueError("factor must be >= 1.0.")

    height, width = frame.shape[:2]
    new_width = int(round(width / factor))
    new_height = int(round(height / factor))
    if new_width < 1 or new_height < 1:
        raise ValueError("factor is too large for the frame size.")

    x0 = (width - new_width) // 2
    y0 = (height - new_height) // 2
    cropped = frame[y0 : y0 + new_height, x0 : x0 + new_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_frame(frame: Frame, width: int, height: int) -> Frame:
    """Resize a frame to the given dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Resize width and height must be positive.")

    frame_height, frame_width = frame.shape[:2]
    if width < frame_width or height < frame_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(frame, (width, height), interpolation=interpolation)


def rotate_frame(frame: Frame, degrees: int) -> Frame:
    """Rotate a frame by 0, 90, 180, or 270 degrees."""
    if degrees == 0:
        return frame
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError("degrees must be one of 0, 90, 180, 270.")


def flip_frame(frame: Frame, direction: str) -> Frame:
    """Flip a frame horizontally, vertically, or both."""
    if direction == "horizontal":
        flip_code = 1
    elif direction == "vertical":
        flip_code = 0
    elif direction == "both":
        flip_code = -1
    else:
        raise ValueError("direction must be 'horizontal', 'vertical', or 'both'.")

    return cv2.flip(frame, flip_code)


def to_grayscale(frame: Frame) -> Frame:
    """Convert a frame to grayscale."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def adjust_brightness_contrast(
    frame: Frame, brightness: float = 0.0, contrast: float = 1.0
) -> Frame:
    """Adjust brightness and contrast."""
    if contrast <= 0:
        raise ValueError("contrast must be positive.")
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def adjust_saturation(frame: Frame, factor: float) -> Frame:
    """Adjust saturation by a factor."""
    if factor < 0:
        raise ValueError("factor must be non-negative.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Saturation adjustment requires a 3-channel BGR frame.")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_gamma(frame: Frame, gamma: float) -> Frame:
    """Adjust gamma using a lookup table."""
    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(frame, table)


def sharpen_frame(frame: Frame, amount: float = 0.5, sigma: float = 1.0) -> Frame:
    """Sharpen a frame using unsharp masking."""
    if amount < 0:
        raise ValueError("amount must be >= 0.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)


def blur_frame(frame: Frame, ksize: int = 5) -> Frame:
    """Blur a frame using a Gaussian kernel."""
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def capture_photo(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    *,
    output_path: Path | None = None,
    output_dir: Path = PHOTO_DIR,
    prefix: str = "photo_",
    suffix: str = ".jpg",
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    processing_steps: Sequence[FrameProcessor] | None = None,
) -> Path:
    """Capture a photo and save it to disk."""
    capture = open_camera(camera_index, width=width, height=height, fps=fps)
    try:
        frame = capture_frame(capture)
    finally:
        capture.release()

    if processing_steps:
        frame = apply_pipeline(frame, processing_steps)

    if output_path is None:
        output_path = auto_increment_path(output_dir, prefix, suffix)

    save_image(frame, output_path)
    return output_path


def record_video(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    *,
    output_path: Path | None = None,
    output_dir: Path = VIDEO_DIR,
    prefix: str = "video_",
    suffix: str = ".mp4",
    duration_seconds: float,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    codec: str = "mp4v",
    processing_steps: Sequence[FrameProcessor] | None = None,
) -> Path:
    """Record a video from the camera."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive.")
    if codec and len(codec) != 4:
        raise ValueError("codec must be a 4-character code.")

    capture = open_camera(camera_index, width=width, height=height, fps=fps)
    try:
        actual_fps = fps if fps is not None else capture.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            raise RuntimeError("Camera FPS is unavailable; pass fps= to record_video.")

        first_frame = capture_frame(capture)
        if processing_steps:
            first_frame = apply_pipeline(first_frame, processing_steps)
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise RuntimeError("Video recording requires a 3-channel BGR frame.")

        frame_height, frame_width = first_frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError("Camera frame size is unavailable; pass width/height.")

        if output_path is None:
            output_path = auto_increment_path(output_dir, prefix, suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, float(actual_fps), (frame_width, frame_height)
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer.")

        try:
            writer.write(first_frame)
            deadline = time.monotonic() + duration_seconds
            while time.monotonic() < deadline:
                frame = capture_frame(capture)
                if processing_steps:
                    frame = apply_pipeline(frame, processing_steps)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise RuntimeError("Video recording requires a 3-channel BGR frame.")
                if frame.shape[:2] != (frame_height, frame_width):
                    raise RuntimeError("Processed frame size does not match writer size.")
                writer.write(frame)
        finally:
            writer.release()
    finally:
        capture.release()

    return output_path


def probe_cameras(max_index: int = 5) -> list[int]:
    """Return camera indexes that can be opened."""
    if max_index < 0:
        raise ValueError("max_index must be >= 0.")

    available: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            available.append(index)
        capture.release()

    return available
