from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google.adk.tools.tool_context import ToolContext

from camera_tools import DEFAULT_CAMERA_INDEX, capture_photo
from niryo_tools import (
    DEFAULT_MOVE_STEP_M,
    calibrate_auto,
    get_joints,
    get_pose,
    move_backward,
    move_down,
    move_forward,
    move_left,
    move_right,
    move_up,
    rotate_yaw,
    robot_session,
    set_arm_speed,
)

from .arm_observations import append_observation, read_observations

DEFAULT_ARM_STEP_M = max(DEFAULT_MOVE_STEP_M, 0.03)
DEFAULT_ARM_SPEED = 30
DEFAULT_ARM_IP = "10.10.10.10"
DEFAULT_ARM_RUN_DIR = Path("captures") / "arm_experiments"
DEFAULT_CAPTURE_RETRIES = 3
DEFAULT_CAPTURE_RETRY_DELAY_S = 0.2
DEFAULT_CAPTURE_WARMUP_FRAMES = 10
DEFAULT_CAPTURE_WARMUP_DELAY_S = 0.1


def _resolve_ip(ip: str | None) -> str:
    """Resolve the robot IP address."""
    return ip or DEFAULT_ARM_IP


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")


def _resolve_run_dir(run_name: str | None) -> Path:
    """Resolve the experiment run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(run_name) if run_name else ""
    folder = f"{timestamp}_{slug}" if slug else timestamp
    run_dir = DEFAULT_ARM_RUN_DIR / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_observation_json(observation_json: str) -> dict[str, Any]:
    """Parse observation JSON."""
    try:
        payload = json.loads(observation_json)
    except json.JSONDecodeError as exc:
        raise ValueError("observation_json must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("observation_json must be a JSON object.")
    return payload


def _capture_with_retries(
    *,
    run_dir: Path,
    prefix: str,
    retries: int = DEFAULT_CAPTURE_RETRIES,
    delay_s: float = DEFAULT_CAPTURE_RETRY_DELAY_S,
    warmup_frames: int = DEFAULT_CAPTURE_WARMUP_FRAMES,
    warmup_delay_s: float = DEFAULT_CAPTURE_WARMUP_DELAY_S,
) -> str:
    """Capture a photo with retries."""
    if retries < 1:
        raise ValueError("retries must be >= 1.")
    if delay_s < 0:
        raise ValueError("delay_s must be >= 0.")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return str(
                capture_photo(
                    camera_index=DEFAULT_CAMERA_INDEX,
                    output_dir=run_dir,
                    prefix=prefix,
                    warmup_frames=warmup_frames,
                    warmup_delay_s=warmup_delay_s,
                )
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries and delay_s:
                time.sleep(delay_s)
    raise RuntimeError("Failed to capture photo after retries.") from last_error


def arm_get_pose(ip: str | None = None) -> dict[str, object]:
    """Get the current robot pose."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_get_joints(ip: str | None = None) -> dict[str, object]:
    """Get the current robot joints."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        joints = get_joints(robot)
    return {"status": "success", "joints": joints}


def arm_calibrate_auto(ip: str | None = None) -> dict[str, object]:
    """Auto-calibrate the robot."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        calibrate_auto(robot)
    return {"status": "success"}


def arm_move_up(step_m: float = DEFAULT_ARM_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot up."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_up(robot, step_m)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_move_down(
    step_m: float = DEFAULT_ARM_STEP_M,
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot down."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_down(robot, step_m)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_move_left(
    step_m: float = DEFAULT_ARM_STEP_M,
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot left."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_left(robot, step_m)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_move_right(
    step_m: float = DEFAULT_ARM_STEP_M,
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot right."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_right(robot, step_m)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_move_forward(
    step_m: float = DEFAULT_ARM_STEP_M,
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot forward."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_forward(robot, step_m)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_move_backward(
    step_m: float = DEFAULT_ARM_STEP_M,
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot backward."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_backward(robot, step_m)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_rotate_yaw(
    angle_rad: float,
    ip: str | None = None,
) -> dict[str, object]:
    """Rotate the robot around yaw."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        rotate_yaw(robot, angle_rad)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def append_arm_observation(
    observation_json: str,
    output_path: str | None = None,
) -> dict[str, object]:
    """Append an observation JSON record."""
    payload = _parse_observation_json(observation_json)
    path = append_observation(payload, output_path=output_path)
    return {"status": "success", "path": str(path)}


def read_arm_observations(
    limit: int = 50,
    output_path: str | None = None,
) -> dict[str, object]:
    """Read recent observation records."""
    records = read_observations(limit=limit, output_path=output_path)
    return {"status": "success", "count": len(records), "observations": records}


def run_arm_experiment(
    run_name: str | None = None,
    *,
    step_m: float = DEFAULT_ARM_STEP_M,
    speed: int = DEFAULT_ARM_SPEED,
    ip: str | None = None,
    capture_images: bool = True,
    observation_path: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Run a basic arm experiment with camera captures."""
    if step_m <= 0:
        raise ValueError("step_m must be > 0.")
    if speed < 1 or speed > 100:
        raise ValueError("speed must be between 1 and 100.")

    run_dir = _resolve_run_dir(run_name)
    observations: list[dict[str, object]] = []

    with robot_session(_resolve_ip(ip), auto_calibrate=True) as robot:
        set_arm_speed(robot, speed)
        base_pose = get_pose(robot)
        base_joints = get_joints(robot)
        base_image = None
        if capture_images:
            base_image = _capture_with_retries(
                run_dir=run_dir,
                prefix="step_00_base_",
            )

        steps = [
            ("move_up", lambda: move_up(robot, step_m)),
            ("move_down", lambda: move_down(robot, step_m)),
            ("move_left", lambda: move_left(robot, step_m)),
            ("move_right", lambda: move_right(robot, step_m)),
            ("move_forward", lambda: move_forward(robot, step_m)),
            ("move_backward", lambda: move_backward(robot, step_m)),
        ]

        for index, (label, action) in enumerate(steps, start=1):
            pose_before = get_pose(robot)
            joints_before = get_joints(robot)
            action()
            pose_after = get_pose(robot)
            joints_after = get_joints(robot)

            image_path = None
            if capture_images:
                image_path = _capture_with_retries(
                    run_dir=run_dir,
                    prefix=f"step_{index:02d}_{label}_",
                )

            record = {
                "run_dir": str(run_dir),
                "step": index,
                "action": label,
                "step_m": step_m,
                "pose_before": pose_before,
                "pose_after": pose_after,
                "joints_before": joints_before,
                "joints_after": joints_after,
                "image_path": image_path,
            }
            append_observation(record, output_path=observation_path)
            observations.append(record)

    if tool_context is not None:
        tool_context.state["last_arm_run_dir"] = str(run_dir)
        tool_context.state["last_arm_base_pose"] = base_pose
        tool_context.state["last_arm_base_joints"] = base_joints
        if base_image:
            tool_context.state["last_arm_base_image"] = base_image

    return {
        "status": "success",
        "run_dir": str(run_dir),
        "base_pose": base_pose,
        "base_joints": base_joints,
        "base_image": base_image,
        "observations": observations,
    }
