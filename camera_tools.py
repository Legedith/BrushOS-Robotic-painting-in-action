from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

Frame = np.ndarray
FrameProcessor = Callable[[Frame], Frame]

DEFAULT_CAMERA_INDEX = 2
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
CAPTURE_ROOT = Path("captures")
PHOTO_DIR = CAPTURE_ROOT / "photos"
VIDEO_DIR = CAPTURE_ROOT / "videos"
DEFAULT_ENHANCE_TARGET_LUMA = 150.0
DEFAULT_ENHANCE_CLIP_LIMIT = 2.0
DEFAULT_ENHANCE_TILE_GRID = 8
DEFAULT_ENHANCE_GAMMA_MIN = 0.4
DEFAULT_ENHANCE_GAMMA_MAX = 2.2
DEFAULT_ENHANCE_SHARPEN_AMOUNT = 0.6
DEFAULT_ENHANCE_SHARPEN_SIGMA = 1.2
DEFAULT_PHOTO_AUTO_ENHANCE = False
DEFAULT_CAMERA_WARMUP_FRAMES = 5
AUTO_CURRENT_LAB_MEAN = (96.1775, 128.4225, 128.8904)
AUTO_CURRENT_LAB_STD = (36.4013, 8.7677, 8.5414)
AUTO_CURRENT_REFERENCE_PATH = Path("captures/presets/20260208_110058/auto_current.jpg")
COLOR_PROFILE_LAB_STATS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "auto_current_20260208_110058": (AUTO_CURRENT_LAB_MEAN, AUTO_CURRENT_LAB_STD),
}
DEFAULT_PHOTO_COLOR_PROFILE = "auto_current_hist"
CAMERA_PROPERTY_IDS: dict[str, int] = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "gamma": cv2.CAP_PROP_GAMMA,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
}


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


def _validate_frame_size(frame: Frame, width: int | None, height: int | None) -> None:
    """Validate the captured frame size against expectations."""
    actual_height, actual_width = frame.shape[:2]
    if width is not None and actual_width != width:
        raise RuntimeError(
            f"Frame width mismatch: requested {width}, got {actual_width}."
        )
    if height is not None and actual_height != height:
        raise RuntimeError(
            f"Frame height mismatch: requested {height}, got {actual_height}."
        )


def set_camera_properties(
    capture: cv2.VideoCapture,
    settings: dict[str, float],
    *,
    tolerance: float = 1.0,
    per_property_tolerance: dict[str, float] | None = None,
) -> dict[str, float]:
    """Set camera properties and return the actual values."""
    actual: dict[str, float] = {}
    for name, value in settings.items():
        if name not in CAMERA_PROPERTY_IDS:
            raise ValueError(f"Unsupported camera property: {name}.")
        if not capture.set(CAMERA_PROPERTY_IDS[name], float(value)):
            raise RuntimeError(f"Failed to set camera property {name}.")
        current = float(capture.get(CAMERA_PROPERTY_IDS[name]))
        if current <= -1:
            raise RuntimeError(f"Camera property {name} is unavailable.")
        actual[name] = current
        if per_property_tolerance:
            property_tolerance = per_property_tolerance.get(name, tolerance)
        else:
            property_tolerance = tolerance
        if property_tolerance >= 0 and abs(current - float(value)) > property_tolerance:
            raise RuntimeError(
                f"Camera property {name} mismatch: requested {value}, got {current:.2f}."
            )
    return actual


def get_camera_properties(capture: cv2.VideoCapture, names: Sequence[str]) -> dict[str, float]:
    """Read camera property values."""
    values: dict[str, float] = {}
    for name in names:
        if name not in CAMERA_PROPERTY_IDS:
            raise ValueError(f"Unsupported camera property: {name}.")
        values[name] = float(capture.get(CAMERA_PROPERTY_IDS[name]))
    return values


def open_camera(
    camera_index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    backend: int | None = None,
    fourcc: str | None = None,
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
        if fourcc is not None:
            if len(fourcc) != 4:
                raise ValueError("fourcc must be a 4-character code.")
            fourcc_value = cv2.VideoWriter_fourcc(*fourcc)
            if not capture.set(cv2.CAP_PROP_FOURCC, float(fourcc_value)):
                raise RuntimeError("Failed to set camera FOURCC.")
            actual = int(capture.get(cv2.CAP_PROP_FOURCC))
            if actual != fourcc_value:
                raise RuntimeError("Camera FOURCC mismatch.")
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


def warmup_capture(
    capture: cv2.VideoCapture,
    frames: int = DEFAULT_CAMERA_WARMUP_FRAMES,
    delay_s: float = 0.05,
) -> Frame:
    """Warm up a camera by reading a few frames."""
    if frames < 1:
        raise ValueError("frames must be >= 1.")
    if delay_s < 0:
        raise ValueError("delay_s must be >= 0.")
    frame: Frame | None = None
    ok = False
    for _ in range(frames):
        ok, frame = capture.read()
        if delay_s:
            time.sleep(delay_s)
    if not ok or frame is None:
        raise RuntimeError("Failed to warm up camera.")
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


def center_crop_frame(frame: Frame, width: int, height: int) -> Frame:
    """Crop a frame around the center."""
    if width <= 0 or height <= 0:
        raise ValueError("Crop width and height must be positive.")

    frame_height, frame_width = frame.shape[:2]
    if width > frame_width or height > frame_height:
        raise ValueError("Crop size exceeds frame dimensions.")

    x0 = (frame_width - width) // 2
    y0 = (frame_height - height) // 2
    return frame[y0 : y0 + height, x0 : x0 + width]


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


def skew_frame(frame: Frame, skew_x: float, skew_y: float) -> Frame:
    """Skew a frame by X and Y factors."""
    height, width = frame.shape[:2]
    matrix = np.array([[1.0, skew_x, 0.0], [skew_y, 1.0, 0.0]], dtype=np.float32)
    new_width = int(round(width + abs(skew_x) * height))
    new_height = int(round(height + abs(skew_y) * width))
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Skew results in invalid dimensions.")

    return cv2.warpAffine(frame, matrix, (new_width, new_height))


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


def _gamma_from_target_luma(mean_luma: float, target_luma: float) -> float:
    """Compute gamma to map mean luminance to a target."""
    if mean_luma <= 0 or mean_luma >= 255:
        raise ValueError("mean_luma must be within (0, 255).")
    if target_luma <= 0 or target_luma >= 255:
        raise ValueError("target_luma must be within (0, 255).")
    return math.log(target_luma / 255.0) / math.log(mean_luma / 255.0)


def auto_enhance_frame(
    frame: Frame,
    *,
    target_luma: float = DEFAULT_ENHANCE_TARGET_LUMA,
    clip_limit: float = DEFAULT_ENHANCE_CLIP_LIMIT,
    tile_grid: int = DEFAULT_ENHANCE_TILE_GRID,
    gamma_min: float = DEFAULT_ENHANCE_GAMMA_MIN,
    gamma_max: float = DEFAULT_ENHANCE_GAMMA_MAX,
    sharpen_amount: float = DEFAULT_ENHANCE_SHARPEN_AMOUNT,
    sharpen_sigma: float = DEFAULT_ENHANCE_SHARPEN_SIGMA,
) -> Frame:
    """Enhance brightness and clarity with CLAHE, gamma, and sharpening."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("auto_enhance_frame expects a 3-channel BGR frame.")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive.")
    if tile_grid <= 0:
        raise ValueError("tile_grid must be positive.")
    if gamma_min <= 0 or gamma_max <= 0 or gamma_min > gamma_max:
        raise ValueError("gamma_min and gamma_max must be positive and ordered.")
    if sharpen_amount < 0:
        raise ValueError("sharpen_amount must be >= 0.")
    if sharpen_sigma <= 0:
        raise ValueError("sharpen_sigma must be positive.")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid)
    )
    l_channel = clahe.apply(l_channel)

    mean_luma = float(np.mean(l_channel))
    gamma = _gamma_from_target_luma(mean_luma, target_luma)
    gamma = max(gamma_min, min(gamma_max, gamma))
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    l_channel = cv2.LUT(l_channel, table)

    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    if sharpen_amount == 0:
        return enhanced

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sharpen_sigma)
    return cv2.addWeighted(enhanced, 1 + sharpen_amount, blurred, -sharpen_amount, 0)


def match_lab_profile(
    frame: Frame,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Frame:
    """Match LAB channel mean/std to a reference profile."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("match_lab_profile expects a 3-channel BGR frame.")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std = lab.reshape(-1, 3).std(axis=0)
    if np.any(src_std <= 0):
        raise RuntimeError("Source LAB standard deviation is zero.")

    target_mean = np.array(mean, dtype=np.float32)
    target_std = np.array(std, dtype=np.float32)
    if np.any(target_std <= 0):
        raise ValueError("Reference LAB standard deviation must be positive.")

    lab = (lab - src_mean) * (target_std / src_std) + target_mean
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def match_histogram_channel(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Match a single-channel histogram to a template."""
    source_flat = source.ravel()
    template_flat = template.ravel()

    s_values, bin_idx, s_counts = np.unique(
        source_flat, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template_flat, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    matched = interp_t_values[bin_idx].reshape(source.shape)
    return matched.astype(np.uint8)


def match_lab_histogram(frame: Frame, reference: Frame) -> Frame:
    """Match LAB histograms to a reference image."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("match_lab_histogram expects a 3-channel BGR frame.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("reference must be a 3-channel BGR frame.")

    src_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    matched = np.empty_like(src_lab)
    for channel in range(3):
        matched[:, :, channel] = match_histogram_channel(
            src_lab[:, :, channel], ref_lab[:, :, channel]
        )

    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)


def apply_color_profile(frame: Frame, profile: str | None) -> Frame:
    """Apply a named color profile to a frame."""
    if profile is None:
        return frame
    if profile == "auto_current_hist":
        if not AUTO_CURRENT_REFERENCE_PATH.exists():
            raise FileNotFoundError(
                f"Reference image not found: {AUTO_CURRENT_REFERENCE_PATH}"
            )
        reference = cv2.imread(str(AUTO_CURRENT_REFERENCE_PATH))
        if reference is None:
            raise RuntimeError(
                f"Reference image unreadable: {AUTO_CURRENT_REFERENCE_PATH}"
            )
        return match_lab_histogram(frame, reference)
    if profile not in COLOR_PROFILE_LAB_STATS:
        raise ValueError(f"Unknown color profile: {profile}.")
    mean, std = COLOR_PROFILE_LAB_STATS[profile]
    return match_lab_profile(frame, mean, std)


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
    width: int | None = DEFAULT_FRAME_WIDTH,
    height: int | None = DEFAULT_FRAME_HEIGHT,
    fps: float | None = None,
    color_profile: str | None = DEFAULT_PHOTO_COLOR_PROFILE,
    auto_enhance: bool = DEFAULT_PHOTO_AUTO_ENHANCE,
    warmup_frames: int = DEFAULT_CAMERA_WARMUP_FRAMES,
    warmup_delay_s: float = 0.05,
    processing_steps: Sequence[FrameProcessor] | None = None,
) -> Path:
    """Capture a photo and save it to disk."""
    capture = open_camera(camera_index, width=width, height=height, fps=fps)
    try:
        frame = warmup_capture(capture, warmup_frames, warmup_delay_s)
        _validate_frame_size(frame, width, height)
    finally:
        capture.release()

    frame = apply_color_profile(frame, color_profile)
    if auto_enhance:
        frame = auto_enhance_frame(frame)
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
    width: int | None = DEFAULT_FRAME_WIDTH,
    height: int | None = DEFAULT_FRAME_HEIGHT,
    codec: str = "mp4v",
    auto_enhance: bool = False,
    warmup_frames: int = DEFAULT_CAMERA_WARMUP_FRAMES,
    warmup_delay_s: float = 0.05,
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

        first_frame = warmup_capture(capture, warmup_frames, warmup_delay_s)
        _validate_frame_size(first_frame, width, height)
        if auto_enhance:
            first_frame = auto_enhance_frame(first_frame)
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
                if auto_enhance:
                    frame = auto_enhance_frame(frame)
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
