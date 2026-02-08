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

DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _ensure_api_key() -> str:
    """Get the Gemini API key from the environment."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return api_key


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
    image = _load_image(Path(image_path))
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
    api_key = _ensure_api_key()
    image_file = Path(image_path)
    mime_type = _detect_mime_type(image_file)
    image_data = image_file.read_bytes()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
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
    api_key = _ensure_api_key()
    image_file = Path(image_path)
    mime_type = _detect_mime_type(image_file)
    image_data = image_file.read_bytes()
    question = prompt or "Describe the image in 1-3 sentences."

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
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
    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    if tool_context is not None:
        tool_context.state["last_description_image"] = image_path
        tool_context.state["last_description_model"] = model
        tool_context.state["last_description_text"] = text
    return {"status": "success", "text": text, "model": model}


def verify_target(
    image_path: str,
    target_description: str,
    model: str = DEFAULT_MODEL,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Verify if a target is visible in an image."""
    _validate_model(model)
    api_key = _ensure_api_key()
    image_file = Path(image_path)
    mime_type = _detect_mime_type(image_file)
    image_data = image_file.read_bytes()

    prompt = (
        "Return JSON only. Determine if the target is clearly visible. "
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

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
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
    api_key = _ensure_api_key()
    image_file = Path(image_path)
    mime_type = _detect_mime_type(image_file)
    image_data = image_file.read_bytes()

    prompt = (
        "Pick one visually prominent item to focus on. "
        "Return JSON with target_description (short, specific) and rotation_degrees "
        "(0, 90, 180, or 270) to make the item upright."
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

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
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
    api_key = _ensure_api_key()
    image_file = Path(image_path)
    mime_type = _detect_mime_type(image_file)
    image = _load_image(image_file)
    height, width = image.shape[:2]
    image_data = image_file.read_bytes()

    prompt = (
        "Return JSON only with keys x, y, width, height. "
        "Use pixel coordinates within the image, with origin at top-left. "
        f"Image size is {width}x{height}. "
        f"Target: {target_description}."
    )

    client = genai.Client(api_key=api_key)
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
    response = client.models.generate_content(
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


root_agent = Agent(
    name="camera_agent",
    model=DEFAULT_MODEL,
    output_key="last_response",
    description="Agent that captures and analyzes camera images.",
    instruction=(
        "You can capture photos, transform them, and answer visual questions. "
        "Use get_session_state and set_session_state to read or update session state. "
        "Use search_memory to recall prior sessions, and call add_session_to_memory "
        "after completing a task or when asked to remember. "
        "When a user asks about what is visible, capture a photo first, then analyze it. "
        "Use locate_object to find a region, then crop to focus on that area when needed. "
        "For multi-step edits, create a run folder and use transform_image_with_intermediates "
        "with steps_json (a JSON array of steps). "
        "When asked to focus on an item, do this sequence: "
        "1) create_run_folder, 2) capture_photo_tool with output_dir set to that folder, "
        "3) describe_image to pick a concrete item, "
        "4) locate_object using that item description, "
        "5) transform_image_with_intermediates to crop/rotate/zoom, "
        "6) verify_target on the final crop. "
        "If verify_target says the item is not present, refine the description and retry. "
        "Always return file paths for any generated images. "
        "Context: last_run_dir={last_run_dir?}, last_photo_path={last_photo_path?}."
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
        verify_target,
        locate_object,
        get_session_state,
        set_session_state,
        search_memory,
        add_session_to_memory,
    ],
)
