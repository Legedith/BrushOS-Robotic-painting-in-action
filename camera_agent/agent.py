from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from google import genai
from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .arm_tools import (
    append_arm_observation,
    arm_calibrate_auto,
    arm_close_gripper,
    arm_draw_strokes,
    arm_get_tool_id,
    arm_get_joints,
    arm_get_pose,
    arm_move_backward,
    arm_move_down,
    arm_move_forward,
    arm_move_joints,
    arm_move_left,
    arm_move_pose,
    arm_move_right,
    arm_move_up,
    arm_refill_paint,
    arm_rotate_yaw,
    arm_release_with_tool,
    arm_set_speed,
    arm_grasp_with_tool,
    arm_open_gripper,
    arm_update_tool,
    build_arc_stroke,
    build_bezier_stroke,
    build_circle_stroke,
    build_s_curve_stroke,
    concat_strokes,
    transform_strokes,
    read_arm_observations,
    run_arm_experiment,
)
from camera_tools import (
    auto_increment_path,
    capture_photo,
    center_crop_frame,
    crop_frame,
    flip_frame,
    resize_frame,
    rotate_frame,
    save_image,
    skew_frame,
    zoom_frame,
    adjust_brightness_contrast,
    adjust_gamma,
    adjust_saturation,
    blur_frame,
    sharpen_frame,
    to_grayscale,
)

from .key_rotation import get_api_keys, with_key_rotation

DEFAULT_MODEL = "gemini-3-flash-preview"
CONCISE_TEXT_INSTRUCTION = (
    "Respond concisely in plain text. "
    "Include only the details needed to answer the question. "
    "If uncertain, respond with 'uncertain'."
)


def _ensure_api_key() -> str:
    """Get the Gemini API key from the environment."""
    return get_api_keys()[0]


def _validate_model(model: str) -> None:
    """Validate the model name."""
    if model == "gemini-3-flash-preview":
        return
    if model.startswith("gemini-2.5"):
        return
    raise ValueError("Model must be gemini-3-flash-preview or a gemini-2.5 model.")


def _detect_mime_type(path: Path) -> str:
    """Detect an image mime type from file extension."""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported image extension: {suffix}.")


def _load_image(path: Path) -> Any:
    """Load an image from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _image_error(image_path: str, exc: Exception) -> dict[str, object]:
    """Return a structured image error response."""
    if isinstance(exc, FileNotFoundError):
        error = "image_not_found"
    else:
        error = "image_read_failed"
    return {
        "status": "error",
        "error": error,
        "path": image_path,
        "message": str(exc),
    }


def _resolve_output_path(
    output_path: str | None,
    output_dir: str | None,
    prefix: str,
    suffix: str,
) -> Path:
    """Resolve the output path for a saved image."""
    if output_path:
        return Path(output_path)
    target_dir = Path(output_dir) if output_dir else Path("captures") / "agent"
    return auto_increment_path(target_dir, prefix, suffix)


def _ensure_json_value(value: object, path: str) -> None:
    """Validate a JSON-serializable value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"State key at {path} must be a string.")
            _ensure_json_value(item, f"{path}.{key}" if path else key)
        return
    raise ValueError(f"State value at {path} is not JSON-serializable.")


def _validate_state_payload(payload: dict[str, object]) -> None:
    """Validate that state payload is JSON-serializable."""
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("State keys must be strings.")
        _ensure_json_value(value, key)


def _parse_state_json(state_json: str) -> dict[str, object]:
    """Parse state updates from JSON."""
    try:
        payload = json.loads(state_json)
    except json.JSONDecodeError as exc:
        raise ValueError("state_json must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("state_json must be a JSON object.")
    _validate_state_payload(payload)
    return payload


def _parse_state_keys(keys_json: str) -> list[str]:
    """Parse state keys from JSON."""
    try:
        payload = json.loads(keys_json)
    except json.JSONDecodeError as exc:
        raise ValueError("keys_json must be valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("keys_json must be a JSON array of strings.")
    keys: list[str] = []
    for key in payload:
        if not isinstance(key, str):
            raise ValueError("keys_json must contain only strings.")
        keys.append(key)
    return keys


def _parse_bbox_json(bbox_json: str, label: str) -> dict[str, float]:
    """Parse a bounding box JSON payload."""
    try:
        payload = json.loads(bbox_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    required = {"x", "y", "width", "height"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"{label} must contain x, y, width, height.")
    width = float(payload["width"])
    height = float(payload["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} width/height must be positive.")
    x = float(payload["x"])
    y = float(payload["y"])
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "center_x": x + width / 2,
        "center_y": y + height / 2,
    }


def _append_state_list(
    tool_context: ToolContext,
    key: str,
    value: object,
    max_items: int,
) -> None:
    """Append a value to a list in session state."""
    existing = tool_context.state.get(key)
    if existing is None:
        items: list[object] = []
    elif isinstance(existing, list):
        items = existing
    else:
        raise ValueError(f"State key {key} must be a list.")
    items.append(value)
    if len(items) > max_items:
        items = items[-max_items:]
    tool_context.state[key] = items


def create_run_folder(
    run_name: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Create a run folder for intermediate images."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = ""
    if run_name:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", run_name.strip()).strip("_")
    folder_name = f"{timestamp}_{slug}" if slug else timestamp
    run_dir = Path("captures") / "agent_runs" / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if tool_context is not None:
        tool_context.state["last_run_dir"] = str(run_dir)
        _append_state_list(tool_context, "recent_runs", str(run_dir), 5)
    return {"status": "success", "path": str(run_dir)}


def capture_photo_tool(
    output_dir: str | None = None,
    prefix: str = "photo_",
    suffix: str = ".jpg",
    color_profile: str | None = None,
    auto_enhance: bool | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Capture a photo from the camera."""
    extra: dict[str, object] = {}
    if output_dir is None and tool_context is not None:
        last_run_dir = tool_context.state.get("last_run_dir")
        if isinstance(last_run_dir, str) and last_run_dir:
            output_dir = last_run_dir
    if output_dir is not None:
        extra["output_dir"] = Path(output_dir)
    if color_profile is not None:
        extra["color_profile"] = color_profile
    if auto_enhance is not None:
        extra["auto_enhance"] = auto_enhance

    output_path = capture_photo(prefix=prefix, suffix=suffix, **extra)
    image = _load_image(Path(output_path))
    height, width = image.shape[:2]
    if tool_context is not None:
        tool_context.state["last_photo_path"] = str(output_path)
        tool_context.state["last_image_path"] = str(output_path)
        tool_context.state["last_photo_size"] = {"width": width, "height": height}
        _append_state_list(tool_context, "recent_photos", str(output_path), 5)
    return {
        "status": "success",
        "path": str(output_path),
        "width": width,
        "height": height,
    }


def image_info(image_path: str) -> dict[str, object]:
    """Read image size and channels."""
    try:
        image = _load_image(Path(image_path))
    except Exception as exc:  # noqa: BLE001
        return _image_error(image_path, exc)
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "status": "success",
        "path": image_path,
        "width": width,
        "height": height,
        "channels": channels,
    }


def crop_image(
    image_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "crop_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Crop an image by coordinates."""
    image = _load_image(Path(image_path))
    cropped = crop_frame(image, x, y, width, height)
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(cropped, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def center_crop_image(
    image_path: str,
    width: int,
    height: int,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "center_crop_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Center-crop an image."""
    image = _load_image(Path(image_path))
    cropped = center_crop_frame(image, width, height)
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(cropped, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def zoom_image(
    image_path: str,
    factor: float,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "zoom_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Zoom an image by a factor."""
    image = _load_image(Path(image_path))
    zoomed = zoom_frame(image, factor)
    height, width = zoomed.shape[:2]
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(zoomed, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def resize_image(
    image_path: str,
    width: int,
    height: int,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "resize_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Resize an image to the given dimensions."""
    image = _load_image(Path(image_path))
    resized = resize_frame(image, width, height)
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(resized, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def rotate_image(
    image_path: str,
    degrees: int,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "rotate_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Rotate an image by 0/90/180/270 degrees."""
    image = _load_image(Path(image_path))
    rotated = rotate_frame(image, degrees)
    height, width = rotated.shape[:2]
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(rotated, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def flip_image(
    image_path: str,
    direction: str,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "flip_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Flip an image horizontally, vertically, or both."""
    image = _load_image(Path(image_path))
    flipped = flip_frame(image, direction)
    height, width = flipped.shape[:2]
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(flipped, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def skew_image(
    image_path: str,
    skew_x: float,
    skew_y: float,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "skew_",
    suffix: str = ".jpg",
) -> dict[str, object]:
    """Skew an image by X and Y factors."""
    image = _load_image(Path(image_path))
    skewed = skew_frame(image, skew_x, skew_y)
    height, width = skewed.shape[:2]
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(skewed, resolved)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def _parse_transform_steps(steps_json: str) -> list[dict[str, object]]:
    """Parse transform steps from JSON."""
    try:
        payload = json.loads(steps_json)
    except json.JSONDecodeError as exc:
        raise ValueError("steps_json must be valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("steps_json must be a JSON array.")
    steps: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each step must be a JSON object.")
        steps.append(item)
    return steps


def _normalize_op(op: str) -> str:
    """Normalize operation names."""
    normalized = op.strip().lower()
    if normalized.endswith("_image"):
        normalized = normalized[: -len("_image")]
    return normalized


def transform_image(
    image_path: str,
    steps_json: str,
    output_path: str | None = None,
    output_dir: str | None = None,
    prefix: str = "transform_",
    suffix: str = ".jpg",
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Apply a sequence of transforms to an image.

    steps_json must be a JSON array of step objects with an "op" field.
    """
    image = _load_image(Path(image_path))
    output = image

    steps = _parse_transform_steps(steps_json)
    for step in steps:
        op = _normalize_op(str(step.get("op", "")))
        if not op:
            raise ValueError("Transform step missing 'op'.")

        if op == "crop":
            output = crop_frame(
                output,
                int(step["x"]),
                int(step["y"]),
                int(step["width"]),
                int(step["height"]),
            )
            continue
        if op == "center_crop":
            output = center_crop_frame(output, int(step["width"]), int(step["height"]))
            continue
        if op == "zoom":
            output = zoom_frame(output, float(step["factor"]))
            continue
        if op == "resize":
            output = resize_frame(output, int(step["width"]), int(step["height"]))
            continue
        if op == "rotate":
            output = rotate_frame(output, int(step["degrees"]))
            continue
        if op == "flip":
            output = flip_frame(output, str(step["direction"]))
            continue
        if op == "skew":
            output = skew_frame(output, float(step["skew_x"]), float(step["skew_y"]))
            continue
        if op == "grayscale":
            output = to_grayscale(output)
            continue
        if op == "brightness_contrast":
            output = adjust_brightness_contrast(
                output,
                brightness=float(step.get("brightness", 0.0)),
                contrast=float(step.get("contrast", 1.0)),
            )
            continue
        if op == "saturation":
            output = adjust_saturation(output, float(step["factor"]))
            continue
        if op == "gamma":
            output = adjust_gamma(output, float(step["gamma"]))
            continue
        if op == "sharpen":
            output = sharpen_frame(
                output,
                amount=float(step.get("amount", 0.5)),
                sigma=float(step.get("sigma", 1.0)),
            )
            continue
        if op == "blur":
            output = blur_frame(output, int(step.get("ksize", 5)))
            continue

        raise ValueError(f"Unknown transform op: {op}.")

    height, width = output.shape[:2]
    resolved = _resolve_output_path(output_path, output_dir, prefix, suffix)
    save_image(output, resolved)
    if tool_context is not None:
        tool_context.state["last_transform_path"] = str(resolved)
        tool_context.state["last_image_path"] = str(resolved)
        _append_state_list(tool_context, "recent_transforms", str(resolved), 5)
    return {"status": "success", "path": str(resolved), "width": width, "height": height}


def transform_image_with_intermediates(
    image_path: str,
    steps_json: str,
    output_dir: str,
    prefix: str = "step",
    suffix: str = ".jpg",
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Apply transforms and save intermediate results.

    Valid ops: crop, center_crop, zoom, resize, rotate, flip, skew, grayscale,
    brightness_contrast, saturation, gamma, sharpen, blur. The agent may also
    use *_image aliases like rotate_image.
    """
    image = _load_image(Path(image_path))
    steps = _parse_transform_steps(steps_json)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    output = image
    results: list[dict[str, object]] = []

    for index, step in enumerate(steps, start=1):
        op = _normalize_op(str(step.get("op", "")))
        if not op:
            raise ValueError("Transform step missing 'op'.")

        if op == "crop":
            output = crop_frame(
                output,
                int(step["x"]),
                int(step["y"]),
                int(step["width"]),
                int(step["height"]),
            )
        elif op == "center_crop":
            output = center_crop_frame(output, int(step["width"]), int(step["height"]))
        elif op == "zoom":
            output = zoom_frame(output, float(step["factor"]))
        elif op == "resize":
            output = resize_frame(output, int(step["width"]), int(step["height"]))
        elif op == "rotate":
            output = rotate_frame(output, int(step["degrees"]))
        elif op == "flip":
            output = flip_frame(output, str(step["direction"]))
        elif op == "skew":
            output = skew_frame(output, float(step["skew_x"]), float(step["skew_y"]))
        elif op == "grayscale":
            output = to_grayscale(output)
        elif op == "brightness_contrast":
            output = adjust_brightness_contrast(
                output,
                brightness=float(step.get("brightness", 0.0)),
                contrast=float(step.get("contrast", 1.0)),
            )
        elif op == "saturation":
            output = adjust_saturation(output, float(step["factor"]))
        elif op == "gamma":
            output = adjust_gamma(output, float(step["gamma"]))
        elif op == "sharpen":
            output = sharpen_frame(
                output,
                amount=float(step.get("amount", 0.5)),
                sigma=float(step.get("sigma", 1.0)),
            )
        elif op == "blur":
            output = blur_frame(output, int(step.get("ksize", 5)))
        else:
            raise ValueError(f"Unknown transform op: {op}.")

        height, width = output.shape[:2]
        filename = f"{prefix}_{index:02d}_{op}{suffix}"
        output_path = run_dir / filename
        save_image(output, output_path)
        results.append(
            {
                "step": index,
                "op": op,
                "path": str(output_path),
                "width": width,
                "height": height,
            }
        )

    final_path = results[-1]["path"] if results else None
    if tool_context is not None:
        tool_context.state["last_run_dir"] = str(run_dir)
        if final_path is not None:
            tool_context.state["last_transform_path"] = str(final_path)
            tool_context.state["last_image_path"] = str(final_path)
            _append_state_list(tool_context, "recent_transforms", str(final_path), 5)
    return {"status": "success", "steps": results, "final_path": final_path}


def analyze_image(
    image_path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Analyze an image using Gemini."""
    _validate_model(model)
    keys = get_api_keys()
    image_file = Path(image_path)
    try:
        mime_type = _detect_mime_type(image_file)
        image_data = image_file.read_bytes()
        image = _load_image(image_file)
    except Exception as exc:  # noqa: BLE001
        return _image_error(image_path, exc)
    height, width = image.shape[:2]

    prompt = (
        f"{CONCISE_TEXT_INSTRUCTION} "
        f"Image size: {width}x{height}. "
        f"Question: {question}"
    )
    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    ],
                )
            ],
        )

    response = with_key_rotation(_call, keys)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    if tool_context is not None:
        tool_context.state["last_analysis_image"] = image_path
        tool_context.state["last_analysis_model"] = model
        tool_context.state["last_analysis_text"] = text
    return {"status": "success", "text": text, "model": model}


def describe_image(
    image_path: str,
    prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Describe an image using Gemini."""
    _validate_model(model)
    keys = get_api_keys()
    image_file = Path(image_path)
    try:
        mime_type = _detect_mime_type(image_file)
        image_data = image_file.read_bytes()
        image = _load_image(image_file)
    except Exception as exc:  # noqa: BLE001
        return _image_error(image_path, exc)
    height, width = image.shape[:2]
    question = prompt or "Describe the image in 1-2 short sentences."
    question = (
        f"{CONCISE_TEXT_INSTRUCTION} "
        f"Image size: {width}x{height}. "
        f"{question}"
    )

    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=question),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    ],
                )
            ],
        )

    response = with_key_rotation(_call, keys)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    if tool_context is not None:
        tool_context.state["last_description_image"] = image_path
        tool_context.state["last_description_model"] = model
        tool_context.state["last_description_text"] = text
    return {"status": "success", "text": text, "model": model}


def compare_images(
    previous_image_path: str,
    current_image_path: str,
    prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Compare two images and describe differences."""
    _validate_model(model)
    keys = get_api_keys()
    previous_file = Path(previous_image_path)
    current_file = Path(current_image_path)
    try:
        previous_mime = _detect_mime_type(previous_file)
        current_mime = _detect_mime_type(current_file)
        previous_data = previous_file.read_bytes()
        current_data = current_file.read_bytes()
        previous_image = _load_image(previous_file)
        current_image = _load_image(current_file)
    except Exception as exc:  # noqa: BLE001
        error = "image_not_found" if isinstance(exc, FileNotFoundError) else "image_read_failed"
        return {
            "status": "error",
            "error": error,
            "previous_path": previous_image_path,
            "current_path": current_image_path,
            "message": str(exc),
        }
    prev_height, prev_width = previous_image.shape[:2]
    curr_height, curr_width = current_image.shape[:2]

    question = prompt or (
        "Compare Image A (previous) and Image B (current). "
        "Focus on the robot gripper and the orange object. "
        "Describe how the gripper moved relative to the orange object, "
        "and whether it is closer, farther, or covering."
    )
    question = (
        "Respond in plain text. "
        f"Image A size: {prev_width}x{prev_height}. "
        f"Image B size: {curr_width}x{curr_height}. "
        f"{question}"
    )

    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=question),
                        types.Part.from_bytes(data=previous_data, mime_type=previous_mime),
                        types.Part.from_bytes(data=current_data, mime_type=current_mime),
                    ],
                )
            ],
        )

    response = with_key_rotation(_call, keys)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    if tool_context is not None:
        tool_context.state["last_compare_previous"] = previous_image_path
        tool_context.state["last_compare_current"] = current_image_path
        tool_context.state["last_compare_model"] = model
        tool_context.state["last_compare_text"] = text
    return {"status": "success", "text": text, "model": model}


def verify_target(
    image_path: str,
    target_description: str,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Verify if a target is visible in an image."""
    _validate_model(model)
    keys = get_api_keys()
    image_file = Path(image_path)
    try:
        mime_type = _detect_mime_type(image_file)
        image_data = image_file.read_bytes()
    except Exception as exc:  # noqa: BLE001
        return _image_error(image_path, exc)

    prompt = (
        "Return JSON only. Determine if the target is clearly visible. "
        "Keep notes under 12 words. "
        f"Target: {target_description}."
    )
    response_config = types.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema={
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "confidence": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": ["present", "confidence", "notes"],
        },
        temperature=0,
    )

    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    ],
                )
            ],
            config=response_config,
        )

    response = with_key_rotation(_call, keys)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    payload = json.loads(text)
    if tool_context is not None:
        tool_context.state["last_verification_image"] = image_path
        tool_context.state["last_verification_target"] = target_description
        tool_context.state["last_verification_model"] = model
        tool_context.state["last_verification_result"] = {
            "present": bool(payload["present"]),
            "confidence": float(payload["confidence"]),
            "notes": str(payload["notes"]),
        }
    return {
        "status": "success",
        "present": bool(payload["present"]),
        "confidence": float(payload["confidence"]),
        "notes": str(payload["notes"]),
        "model": model,
    }


def get_session_state(
    keys_json: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Read session state keys or full state."""
    if tool_context is None:
        raise RuntimeError("tool_context is required.")
    state = tool_context.state.to_dict()
    missing: list[str] = []
    if keys_json:
        keys = _parse_state_keys(keys_json)
        selected = {key: state[key] for key in keys if key in state}
        missing = [key for key in keys if key not in state]
    else:
        selected = state
    _validate_state_payload(selected)
    return {"status": "success", "state": selected, "missing": missing}


def set_session_state(
    state_json: str,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Update session state from JSON."""
    if tool_context is None:
        raise RuntimeError("tool_context is required.")
    payload = _parse_state_json(state_json)
    tool_context.state.update(payload)
    return {"status": "success", "updated_keys": sorted(payload.keys())}


def _content_text(content: types.Content | None) -> str:
    """Extract concatenated text from a content object."""
    if not content or not content.parts:
        return ""
    texts = [part.text for part in content.parts if part.text]
    return "\n".join(texts).strip()


async def search_memory(
    query: str,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Search long-term memory for a query."""
    if tool_context is None:
        raise RuntimeError("tool_context is required.")
    response = await tool_context.search_memory(query)
    memories: list[dict[str, object]] = []
    for memory in response.memories:
        memories.append(
            {
                "text": _content_text(memory.content),
                "author": memory.author,
                "timestamp": memory.timestamp,
                "custom_metadata": memory.custom_metadata,
                "id": memory.id,
            }
        )
    return {"status": "success", "count": len(memories), "memories": memories}


async def add_session_to_memory(
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Add the current session to memory."""
    if tool_context is None:
        raise RuntimeError("tool_context is required.")
    await tool_context.add_session_to_memory()
    return {"status": "success"}


def _select_target_from_image(
    image_path: str,
    model: str,
) -> dict[str, object]:
    """Select a target item and rotation for an image."""
    _validate_model(model)
    keys = get_api_keys()
    image_file = Path(image_path)
    mime_type = _detect_mime_type(image_file)
    image_data = image_file.read_bytes()
    image = _load_image(image_file)
    height, width = image.shape[:2]

    prompt = (
        "Pick one visually prominent item to focus on. "
        "Return JSON only with target_description (3-8 words, specific) and "
        "rotation_degrees (0, 90, 180, or 270) to make the item upright. "
        f"Image size is {width}x{height}."
    )
    response_config = types.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema={
            "type": "object",
            "properties": {
                "target_description": {"type": "string"},
                "rotation_degrees": {"type": "integer"},
            },
            "required": ["target_description", "rotation_degrees"],
        },
        temperature=0,
    )

    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    ],
                )
            ],
            config=response_config,
        )

    response = with_key_rotation(_call, keys)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    payload = json.loads(text)
    return {
        "target_description": str(payload["target_description"]),
        "rotation_degrees": int(payload["rotation_degrees"]),
    }


def focus_item_workflow(
    run_name: str | None = None,
    target_hint: str | None = None,
    zoom_factor: float = 1.4,
    padding: float = 0.1,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Capture, focus, and verify a single item with intermediates."""
    _validate_model(model)
    run_dir = Path(create_run_folder(run_name or "focus_item", tool_context=tool_context)["path"])

    capture = capture_photo_tool(
        output_dir=str(run_dir),
        prefix="step_00_raw_",
        tool_context=tool_context,
    )
    raw_path = capture["path"]

    target_info = {"target_description": target_hint, "rotation_degrees": 0}
    if not target_hint:
        target_info = _select_target_from_image(raw_path, model)

    image = _load_image(Path(raw_path))
    rotation = int(target_info["rotation_degrees"])
    if rotation:
        image = rotate_frame(image, rotation)
        rotated_path = run_dir / "step_01_rotated.jpg"
        save_image(image, rotated_path)
        working_path = rotated_path
    else:
        working_path = Path(raw_path)

    target_description = str(target_info["target_description"])
    bbox = locate_object(
        str(working_path),
        target_description,
        model=model,
        tool_context=tool_context,
    )

    def crop_with_padding(pad: float, suffix: str) -> Path:
        image_local = _load_image(working_path)
        height, width = image_local.shape[:2]
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        pad_w = int(w * pad)
        pad_h = int(h * pad)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(width, x + w + pad_w)
        y1 = min(height, y + h + pad_h)
        cropped = crop_frame(image_local, x0, y0, x1 - x0, y1 - y0)
        crop_path = run_dir / f"step_02_crop{suffix}.jpg"
        save_image(cropped, crop_path)
        zoomed = zoom_frame(cropped, zoom_factor)
        zoom_path = run_dir / f"step_03_zoom{suffix}.jpg"
        save_image(zoomed, zoom_path)
        return zoom_path

    attempts = 2
    final_path = crop_with_padding(padding, "")
    verification = verify_target(
        str(final_path),
        target_description,
        model=model,
        tool_context=tool_context,
    )

    while attempts > 0 and not verification["present"]:
        padding += 0.1
        attempts -= 1
        final_path = crop_with_padding(padding, f"_retry{attempts}")
        verification = verify_target(
            str(final_path),
            target_description,
            model=model,
            tool_context=tool_context,
        )

    if tool_context is not None:
        tool_context.state["last_focus_run_dir"] = str(run_dir)
        tool_context.state["last_focus_target"] = target_description
        tool_context.state["last_focus_path"] = str(final_path)
        tool_context.state["last_focus_verification"] = verification
        _append_state_list(tool_context, "recent_focus_paths", str(final_path), 5)

    return {
        "status": "success",
        "run_dir": str(run_dir),
        "target_description": target_description,
        "final_path": str(final_path),
        "verification": verification,
    }


def locate_object(
    image_path: str,
    target_description: str,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Locate an object and return a bounding box."""
    _validate_model(model)
    keys = get_api_keys()
    image_file = Path(image_path)
    try:
        mime_type = _detect_mime_type(image_file)
        image = _load_image(image_file)
        image_data = image_file.read_bytes()
    except Exception as exc:  # noqa: BLE001
        return _image_error(image_path, exc)
    height, width = image.shape[:2]

    prompt = (
        "Return JSON only with keys x, y, width, height. "
        "Use pixel coordinates within the image, with origin at top-left. "
        "Use a tight bounding box with no padding. "
        f"Image size is {width}x{height}. "
        f"Target: {target_description}."
    )

    response_config = types.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
            },
            "required": ["x", "y", "width", "height"],
        },
        temperature=0,
    )
    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    ],
                )
            ],
            config=response_config,
        )

    response = with_key_rotation(_call, keys)
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    payload = json.loads(text)
    if tool_context is not None:
        tool_context.state["last_bbox"] = {
            "image_path": image_path,
            "target_description": target_description,
            "x": int(payload["x"]),
            "y": int(payload["y"]),
            "width": int(payload["width"]),
            "height": int(payload["height"]),
            "model": model,
        }
    return {
        "status": "success",
        "x": int(payload["x"]),
        "y": int(payload["y"]),
        "width": int(payload["width"]),
        "height": int(payload["height"]),
        "model": model,
    }


def plan_relative_move(
    target_bbox_json: str,
    gripper_bbox_json: str,
    pixel_to_meter: float = 0.0005,
    margin_px: float = 6.0,
    min_step_m: float = 0.01,
    max_step_m: float = 0.07,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Suggest arm moves so the gripper covers a target."""
    if pixel_to_meter <= 0:
        raise ValueError("pixel_to_meter must be > 0.")
    if margin_px < 0:
        raise ValueError("margin_px must be >= 0.")
    if min_step_m <= 0 or max_step_m <= 0 or min_step_m > max_step_m:
        raise ValueError("Invalid min/max step sizes.")

    target = _parse_bbox_json(target_bbox_json, "target_bbox_json")
    gripper = _parse_bbox_json(gripper_bbox_json, "gripper_bbox_json")
    delta_x = target["center_x"] - gripper["center_x"]
    delta_y = target["center_y"] - gripper["center_y"]

    def _step(pixels: float) -> float:
        meters = abs(pixels) * pixel_to_meter
        return max(min_step_m, min(max_step_m, meters))

    moves: list[dict[str, object]] = []
    if abs(delta_x) > margin_px:
        if delta_x > 0:
            moves.append(
                {
                    "tool": "arm_move_right",
                    "step_m": _step(delta_x),
                    "reason": "Target appears to the gripper's right in the image.",
                }
            )
        else:
            moves.append(
                {
                    "tool": "arm_move_left",
                    "step_m": _step(delta_x),
                    "reason": "Target appears to the gripper's left in the image.",
                }
            )
    if abs(delta_y) > margin_px:
        if delta_y > 0:
            moves.append(
                {
                    "tool": "arm_move_down",
                    "step_m": _step(delta_y),
                    "reason": "Target appears lower in the frame; move gripper down.",
                }
            )
        else:
            moves.append(
                {
                    "tool": "arm_move_up",
                    "step_m": _step(delta_y),
                    "reason": "Target appears higher; lift the gripper.",
                }
            )

    if not moves:
        moves.append(
            {
                "tool": "capture_photo_tool",
                "reason": "No significant pixel delta; re-check alignment before moving.",
            }
        )

    response = {
        "status": "success",
        "delta_pixels": {"x": delta_x, "y": delta_y},
        "recommended_moves": moves,
        "notes": (
            "Positive delta_x means the target is right of the gripper. "
            "Positive delta_y means the target is lower in the image. "
            "Verify each recommendation by capturing a new photo."
        ),
    }
    if tool_context is not None:
        tool_context.state["last_move_plan"] = response
    return response


root_agent = Agent(
    name="camera_agent",
    model=DEFAULT_MODEL,
    output_key="last_response",
    description="Agent that captures and analyzes camera images.",
    instruction=(
        "Mission: capture what the camera sees, reason about the scene, and physically move the Niryo "
        "gripper (6-DoF) until it covers the user-specified target. Keep responses concise and focused on the "
        "latest evidence.\n"
        "\n"
        "Capabilities overview:\n"
        "- Camera capture: create_run_folder builds a timestamped workspace, and capture_photo_tool always "
        "returns an on-disk image path plus width/height. All downstream tools must use these exact paths.\n"
        "- Vision reasoning: describe_image/analyze_image summarize a single frame, compare_images contrasts "
        "two captures, locate_object/verify_target/plan_relative_move operate on bounding boxes, and "
        "transform_image*_ helpers crop or zoom when focusing on subregions.\n"
        "- Arm & gripper: arm_move_* apply ±0.03–0.05 m relative shifts, arm_move_pose/joints set absolutes, "
        "arm_open/close_gripper and arm_grasp_with_tool manipulate the tool, and append_arm_observation plus "
        "read_arm_observations persist experiment logs.\n"
        "\n"
        "Coordinate mapping (camera faces the workcell head-on):\n"
        "- Target appears to the LEFT in the image → move the gripper LEFT (arm_move_left = +Y).\n"
        "- Target appears to the RIGHT → move RIGHT (arm_move_right = -Y).\n"
        "- Target appears HIGHER → move UP (arm_move_up = +Z). Target LOWER → move DOWN (-Z).\n"
        "- Target appears FARTHER (smaller/near top of mat) → move FORWARD (+X). Closer/bottom → move "
        "BACKWARD (-X). Yaw rotates the wrist to align the gripper orientation.\n"
        "Always confirm the mapping by comparing consecutive captures. If reality contradicts the mapping, "
        "invert that axis and note it in movement_history.\n"
        "\n"
        "Closed-loop workflow (never skip steps):\n"
        "1) Start a run with create_run_folder (store path in session state) and capture_photo_tool. "
        "Always capture before describing or answering any visual question.\n"
        "2) Use describe_image only to pick concrete labels. Immediately call locate_object at least twice: "
        "once for the target and once for the gripper/tool so you have bounding boxes.\n"
        "3) Decide intent. Prefer plan_relative_move(target_bbox_json, gripper_bbox_json) to convert pixel "
        "offsets into candidate arm_move_* calls. If you reason manually, log intent in session state.\n"
        "4) Execute exactly one motion tool (arm_move_*, arm_move_pose, arm_rotate_yaw, gripper actions). "
        "Keep step_m ≤ 0.05 unless repeated evidence shows small moves are insufficient.\n"
        "5) Immediately capture_photo_tool again using the same run folder. Compare with the previous frame "
        "using compare_images and/or a fresh locate_object call to measure the result.\n"
        "6) Log the outcome via append_arm_observation (include move, pose, photo path, and qualitative "
        "result) and update session state keys: last_move, last_move_intent, last_move_result, "
        "last_image_path, last_photo_path, movement_history (list of dicts with move/intent/result/path). "
        "Repeat steps 3–6 until verify_target confirms the target is covered.\n"
        "\n"
        "Drawing & paint workflow (poster-scale strokes only):\n"
        "- Use arm_refill_paint (bowl_top -> bowl_bottom_1 -> bowl_bottom_2 -> bowl_top) before any drawing "
        "and after every 1-2 strokes. Keep refill_after small; heavy brush needs frequent ink.\n"
        "- Use build_* tools to generate geometry: build_circle_stroke, build_arc_stroke, build_s_curve_stroke, "
        "build_bezier_stroke. Combine with transform_strokes and concat_strokes to scale/rotate/translate.\n"
        "- Use arm_draw_strokes to execute: strokes_json is a JSON list of strokes, each stroke is a list of "
        "[u, v] points normalized to the paper (u=0 left->1 right, v=0 top->1 bottom).\n"
        "- Keep strokes BIG: avoid small details or short segments. Prefer steps=1-3, lift_m around 0.02-0.04, "
        "and long continuous lines. If a stroke is <10% of page width/height, skip it.\n"
        "- Keep a margin: stay within u/v ~0.08 to 0.92 unless the user explicitly wants edge-to-edge ink.\n"
        "- Design guidance (no presets):\n"
        "  * Enso: a large arc (not full circle) with a small intentional gap. Use build_arc_stroke with "
        "radius ~0.30-0.35 and leave a 20-40 degree gap.\n"
        "  * Yin-yang: outer circle + S-curve divider + two smaller circles. The S-curve is an offset pair of "
        "half-circles with radius = outer/2; small circles are radius ~0.18 * outer. Use build_s_curve_stroke "
        "and build_circle_stroke, then translate/scale.\n"
        "  * Calligraphy letters/kanji: build from 3-6 long straight or gently curved strokes; avoid tiny hooks.\n"
        "- After a few strokes, capture_photo_tool and use analyze_image/compare_images to judge coverage. "
        "If the shape is weak, add another bold stroke rather than many small ones.\n"
        "\n"
        "Tool-specific guidance:\n"
        "- Never use describe_image, compare_images, or locate_object on files that were not captured in the "
        "current run. If any tool reports image_not_found, capture a new frame and retry.\n"
        "- plan_relative_move assumes the mapping above and returns a recommended tool + step_m list; follow it "
        "unless observations contradict it, in which case revise the mapping notes.\n"
        "- Use transform_image_with_intermediates (with steps_json arrays) whenever you need a series of "
        "crops/rotations/zooms and keep all intermediates in the run folder.\n"
        "- Before grasping or applying force, call arm_update_tool, arm_open_gripper, and arm_close_gripper as "
        "needed, and capture a verification photo.\n"
        "- run_arm_experiment is available to collect calibration data; review past logs with "
        "read_arm_observations when planning future moves.\n"
        "\n"
        "Recovery & safety rules:\n"
        "- If two moves along the same axis worsen the alignment, reverse that axis or switch to another axis. "
        "If camera feedback flips unexpectedly, assume the axis mapping was wrong and invert your plan.\n"
        "- After every motion, take a new photo before answering the user. Never hallucinate changes.\n"
        "- If verify_target reports the item missing, refine the description, recapture, and repeat the "
        "locate_object → plan → move loop.\n"
        "\n"
        "Memory usage:\n"
        "- Use get_session_state/set_session_state as a scratchpad for run metadata (last_run_dir, "
        "recent_photos, movement_history, last_move_intent/result) and clear or summarize it with "
        "add_session_to_memory when a task completes. search_memory can revive prior run insights.\n"
        "\n"
        "Always return every generated file path (captures, crops, intermediates) plus the run folder path. "
        "Context snapshot: last_run_dir={last_run_dir?}, last_photo_path={last_photo_path?}, "
        "last_move={last_move?}, last_move_intent={last_move_intent?}, last_move_result={last_move_result?}, "
        "last_image_path={last_image_path?}."
    ),
    tools=[
        create_run_folder,
        capture_photo_tool,
        image_info,
        crop_image,
        center_crop_image,
        zoom_image,
        resize_image,
        rotate_image,
        flip_image,
        skew_image,
        transform_image,
        transform_image_with_intermediates,
        focus_item_workflow,
        analyze_image,
        describe_image,
        compare_images,
        verify_target,
        locate_object,
        plan_relative_move,
        arm_get_pose,
        arm_get_joints,
        arm_calibrate_auto,
        arm_set_speed,
        arm_update_tool,
        arm_get_tool_id,
        arm_refill_paint,
        arm_draw_strokes,
        arm_open_gripper,
        arm_close_gripper,
        arm_grasp_with_tool,
        arm_release_with_tool,
        build_circle_stroke,
        build_arc_stroke,
        build_s_curve_stroke,
        build_bezier_stroke,
        transform_strokes,
        concat_strokes,
        arm_move_up,
        arm_move_down,
        arm_move_left,
        arm_move_right,
        arm_move_forward,
        arm_move_backward,
        arm_rotate_yaw,
        arm_move_joints,
        arm_move_pose,
        run_arm_experiment,
        append_arm_observation,
        read_arm_observations,
        get_session_state,
        set_session_state,
        search_memory,
        add_session_to_memory,
    ],
)
