from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from google.adk.tools.tool_context import ToolContext

from camera_tools import DEFAULT_CAMERA_INDEX, capture_photo
from niryo_tools import (
    DEFAULT_MOVE_STEP_M,
    calibrate_auto,
    get_joints,
    get_pose,
    move_joints,
    move_backward,
    move_down,
    move_forward,
    move_left,
    move_right,
    move_pose,
    move_up,
    rotate_yaw,
    robot_session,
    set_arm_speed,
)
from pyniryo import PoseObject
from pyniryo.api.enums_communication import Command

from .arm_observations import append_observation, read_observations

DEFAULT_ARM_STEP_M = max(DEFAULT_MOVE_STEP_M, 0.03)
DEFAULT_ARM_SPEED = 30
DEFAULT_ARM_IP = "10.10.10.10"
DEFAULT_ARM_RUN_DIR = Path("captures") / "arm_experiments"
DEFAULT_CAPTURE_RETRIES = 3
DEFAULT_CAPTURE_RETRY_DELAY_S = 0.2
DEFAULT_CAPTURE_WARMUP_FRAMES = 10
DEFAULT_CAPTURE_WARMUP_DELAY_S = 0.1
SAFE_HOME_JOINTS = (0.0, 0.3, -1.3, 0.0, 0.0, 0.0)
DEFAULT_POSES_FILE = "poses.json"
PAPER_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
BOWL_POSES = ("bowl_top", "bowl_bottom_1", "bowl_bottom_2")
DEFAULT_DRAW_PAUSE_S = 0.15
DEFAULT_DRAW_LIFT_M = 0.03
DEFAULT_DRAW_STEPS = 2
DEFAULT_REFILL_AFTER = 1


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


def _load_poses(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load pose definitions from JSON."""
    poses_path = Path(path)
    with poses_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("poses file must be a JSON object mapping names to poses.")
    return payload


def _pose_to_list(pose_dict: dict[str, Any]) -> list[float]:
    pose = PoseObject.from_dict(pose_dict)
    return pose.to_list()


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_pose(p0: Iterable[float], p1: Iterable[float], t: float) -> list[float]:
    return [_lerp(a, b, t) for a, b in zip(p0, p1)]


def _bilinear_pose(
    top_left: list[float],
    top_right: list[float],
    bottom_left: list[float],
    bottom_right: list[float],
    u: float,
    v: float,
) -> list[float]:
    top = _lerp_pose(top_left, top_right, u)
    bottom = _lerp_pose(bottom_left, bottom_right, u)
    return _lerp_pose(top, bottom, v)


def _lift_pose(pose: list[float], lift_m: float) -> list[float]:
    lifted = list(pose)
    lifted[2] += lift_m
    return lifted


def _move_and_pause(robot, pose: list[float], pause_s: float) -> None:
    move_pose(robot, pose)
    if pause_s:
        time.sleep(pause_s)


def _dip_paint(
    robot,
    bowl_top: list[float],
    bowl_bottom_1: list[float],
    bowl_bottom_2: list[float],
    pause_s: float,
) -> None:
    _move_and_pause(robot, bowl_top, pause_s)
    _move_and_pause(robot, bowl_bottom_1, pause_s)
    _move_and_pause(robot, bowl_bottom_2, pause_s)
    _move_and_pause(robot, bowl_top, pause_s)


def _move_line(
    robot,
    start_pose: list[float],
    end_pose: list[float],
    steps: int,
    pause_s: float,
) -> None:
    if steps < 1:
        steps = 1
    for i in range(1, steps + 1):
        t = i / steps
        pose = _lerp_pose(start_pose, end_pose, t)
        _move_and_pause(robot, pose, pause_s)


def _stroke_polyline(
    robot,
    points: list[tuple[float, float]],
    *,
    tl: list[float],
    tr: list[float],
    bl: list[float],
    br: list[float],
    lift_m: float,
    steps: int,
    pause_s: float,
) -> None:
    if len(points) < 2:
        return

    start_pose = _bilinear_pose(tl, tr, bl, br, points[0][0], points[0][1])
    _move_and_pause(robot, _lift_pose(start_pose, lift_m), pause_s)
    _move_and_pause(robot, start_pose, pause_s)

    for (u0, v0), (u1, v1) in zip(points, points[1:]):
        p0 = _bilinear_pose(tl, tr, bl, br, u0, v0)
        p1 = _bilinear_pose(tl, tr, bl, br, u1, v1)
        _move_line(robot, p0, p1, steps, pause_s)

    end_pose = _bilinear_pose(tl, tr, bl, br, points[-1][0], points[-1][1])
    _move_and_pause(robot, _lift_pose(end_pose, lift_m), pause_s)


def _coerce_strokes(payload: object) -> list[list[tuple[float, float]]]:
    """Normalize strokes payload into a list of polylines."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("strokes_json must be valid JSON.") from exc

    if not isinstance(payload, list):
        raise ValueError("strokes must be a list of strokes.")

    strokes: list[list[tuple[float, float]]] = []
    for stroke in payload:
        if not isinstance(stroke, list):
            raise ValueError("Each stroke must be a list of [u, v] points.")
        points: list[tuple[float, float]] = []
        for point in stroke:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
            ):
                raise ValueError("Each point must be [u, v].")
            u = float(point[0])
            v = float(point[1])
            points.append((u, v))
        strokes.append(points)
    return strokes


def _circle_points(center: tuple[float, float], radius: float, segments: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        u = center[0] + radius * math.cos(angle)
        v = center[1] + radius * math.sin(angle)
        points.append((u, v))
    return points


def _arc_points(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        t = i / segments
        angle = _lerp(start_angle, end_angle, t)
        u = center[0] + radius * math.cos(angle)
        v = center[1] + radius * math.sin(angle)
        points.append((u, v))
    return points


def _s_curve_points(center: tuple[float, float], radius: float, segments: int) -> list[tuple[float, float]]:
    cx, cy = center
    r = radius / 2
    points: list[tuple[float, float]] = []

    for i in range(segments + 1):
        t = i / segments
        angle = math.radians(90 - 180 * t)
        u = cx + r * math.cos(angle)
        v = (cy - r) + r * math.sin(angle)
        points.append((u, v))

    for i in range(segments + 1):
        t = i / segments
        angle = math.radians(90 + 180 * t)
        u = cx + r * math.cos(angle)
        v = (cy + r) + r * math.sin(angle)
        points.append((u, v))
    return points


def _bezier_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    segments: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        t = i / segments
        mt = 1 - t
        u = (
            mt**3 * p0[0]
            + 3 * mt**2 * t * p1[0]
            + 3 * mt * t**2 * p2[0]
            + t**3 * p3[0]
        )
        v = (
            mt**3 * p0[1]
            + 3 * mt**2 * t * p1[1]
            + 3 * mt * t**2 * p2[1]
            + t**3 * p3[1]
        )
        points.append((u, v))
    return points




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


def arm_set_speed(percentage_speed: int, ip: str | None = None) -> dict[str, object]:
    """Set the arm speed percentage."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        set_arm_speed(robot, percentage_speed)
    return {"status": "success"}


def arm_update_tool(ip: str | None = None) -> dict[str, object]:
    """Update the attached tool."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        robot.update_tool()
        tool_id = robot.get_current_tool_id()
    return {"status": "success", "tool_id": str(tool_id)}


def arm_get_tool_id(ip: str | None = None) -> dict[str, object]:
    """Get the current tool ID."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        tool_id = robot.get_current_tool_id()
    return {"status": "success", "tool_id": str(tool_id)}


def arm_open_gripper(
    speed: int = 500,
    max_torque_percentage: int | None = None,
    hold_torque_percentage: int | None = None,
    ip: str | None = None,
) -> dict[str, object]:
    """Open the gripper."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        if max_torque_percentage is None or hold_torque_percentage is None:
            sender = robot._NiryoRobot__send_n_receive
            sender(Command.OPEN_GRIPPER, speed)
        else:
            robot.open_gripper(
                speed=speed,
                max_torque_percentage=max_torque_percentage,
                hold_torque_percentage=hold_torque_percentage,
            )
    return {"status": "success"}


def arm_close_gripper(
    speed: int = 500,
    max_torque_percentage: int | None = None,
    hold_torque_percentage: int | None = None,
    ip: str | None = None,
) -> dict[str, object]:
    """Close the gripper."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        if max_torque_percentage is None or hold_torque_percentage is None:
            sender = robot._NiryoRobot__send_n_receive
            sender(Command.CLOSE_GRIPPER, speed)
        else:
            robot.close_gripper(
                speed=speed,
                max_torque_percentage=max_torque_percentage,
                hold_torque_percentage=hold_torque_percentage,
            )
    return {"status": "success"}


def arm_grasp_with_tool(ip: str | None = None) -> dict[str, object]:
    """Grasp with the equipped tool."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        robot.grasp_with_tool()
    return {"status": "success"}


def arm_release_with_tool(ip: str | None = None) -> dict[str, object]:
    """Release with the equipped tool."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        robot.release_with_tool()
    return {"status": "success"}


def arm_move_joints(
    joints: list[float],
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot to joint positions."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_joints(robot, joints)
        pose = get_pose(robot)
    return {"status": "success", "pose": pose}


def arm_move_pose(
    pose: list[float],
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot to an absolute pose."""
    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        move_pose(robot, pose)
        pose_after = get_pose(robot)
    return {"status": "success", "pose": pose_after}


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
        move_joints(robot, SAFE_HOME_JOINTS)
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


def arm_refill_paint(
    poses_file: str = DEFAULT_POSES_FILE,
    *,
    pause_s: float = DEFAULT_DRAW_PAUSE_S,
    speed: int | None = None,
    ip: str | None = None,
) -> dict[str, object]:
    """Dip the brush at the bowl poses (top -> bottom1 -> bottom2 -> top)."""
    if pause_s < 0:
        raise ValueError("pause_s must be >= 0.")
    if speed is not None and not (1 <= speed <= 100):
        raise ValueError("speed must be between 1 and 100.")

    poses = _load_poses(poses_file)
    missing = [name for name in BOWL_POSES if name not in poses]
    if missing:
        raise ValueError(f"Missing bowl poses in {poses_file}: {', '.join(missing)}")

    bowl_top = _pose_to_list(poses["bowl_top"])
    bowl_bottom_1 = _pose_to_list(poses["bowl_bottom_1"])
    bowl_bottom_2 = _pose_to_list(poses["bowl_bottom_2"])

    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        if speed is not None:
            set_arm_speed(robot, speed)
        _dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, pause_s)

    return {"status": "success", "poses_file": poses_file}


def arm_draw_strokes(
    strokes_json: str,
    *,
    poses_file: str = DEFAULT_POSES_FILE,
    pause_s: float = DEFAULT_DRAW_PAUSE_S,
    lift_m: float = DEFAULT_DRAW_LIFT_M,
    steps: int = DEFAULT_DRAW_STEPS,
    refill_after: int = DEFAULT_REFILL_AFTER,
    start_with_refill: bool = True,
    speed: int | None = None,
    ip: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
    """Draw polylines on the paper using normalized coordinates (JSON string or list)."""
    if pause_s < 0:
        raise ValueError("pause_s must be >= 0.")
    if lift_m < 0:
        raise ValueError("lift_m must be >= 0.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if refill_after < 1:
        raise ValueError("refill_after must be >= 1.")
    if speed is not None and not (1 <= speed <= 100):
        raise ValueError("speed must be between 1 and 100.")

    poses = _load_poses(poses_file)
    missing = [name for name in (*PAPER_CORNERS, *BOWL_POSES) if name not in poses]
    if missing:
        raise ValueError(f"Missing poses in {poses_file}: {', '.join(missing)}")

    strokes = _coerce_strokes(strokes_json)

    tl = _pose_to_list(poses["top_left"])
    tr = _pose_to_list(poses["top_right"])
    bl = _pose_to_list(poses["bottom_left"])
    br = _pose_to_list(poses["bottom_right"])

    bowl_top = _pose_to_list(poses["bowl_top"])
    bowl_bottom_1 = _pose_to_list(poses["bowl_bottom_1"])
    bowl_bottom_2 = _pose_to_list(poses["bowl_bottom_2"])

    with robot_session(_resolve_ip(ip), auto_calibrate=False) as robot:
        if speed is not None:
            set_arm_speed(robot, speed)
        if start_with_refill:
            _dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, pause_s)

        stroke_count = 0
        for stroke in strokes:
            if stroke_count >= refill_after:
                _dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, pause_s)
                stroke_count = 0
            _stroke_polyline(
                robot,
                stroke,
                tl=tl,
                tr=tr,
                bl=bl,
                br=br,
                lift_m=lift_m,
                steps=steps,
                pause_s=pause_s,
            )
            stroke_count += 1

    if tool_context is not None:
        tool_context.state["last_draw_strokes"] = len(strokes)
        tool_context.state["last_draw_poses_file"] = poses_file

    return {
        "status": "success",
        "strokes": len(strokes),
        "poses_file": poses_file,
    }



def build_circle_stroke(
    center_u: float,
    center_v: float,
    radius: float,
    segments: int = 24,
) -> dict[str, object]:
    """Return a circular stroke as normalized [u, v] points."""
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if segments < 3:
        raise ValueError("segments must be >= 3.")
    stroke = _circle_points((center_u, center_v), radius, segments)
    return {"status": "success", "stroke": stroke, "strokes": [stroke]}


def build_arc_stroke(
    center_u: float,
    center_v: float,
    radius: float,
    start_deg: float,
    end_deg: float,
    segments: int = 24,
) -> dict[str, object]:
    """Return an arc stroke as normalized [u, v] points."""
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if segments < 2:
        raise ValueError("segments must be >= 2.")
    stroke = _arc_points(
        (center_u, center_v),
        radius,
        math.radians(start_deg),
        math.radians(end_deg),
        segments,
    )
    return {"status": "success", "stroke": stroke, "strokes": [stroke]}


def build_s_curve_stroke(
    center_u: float,
    center_v: float,
    radius: float,
    segments: int = 24,
) -> dict[str, object]:
    """Return an S-curve (taiji divider) stroke."""
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if segments < 2:
        raise ValueError("segments must be >= 2.")
    stroke = _s_curve_points((center_u, center_v), radius, segments)
    return {"status": "success", "stroke": stroke, "strokes": [stroke]}


def build_bezier_stroke(
    p0_u: float,
    p0_v: float,
    p1_u: float,
    p1_v: float,
    p2_u: float,
    p2_v: float,
    p3_u: float,
    p3_v: float,
    segments: int = 24,
) -> dict[str, object]:
    """Return a cubic Bezier stroke."""
    if segments < 2:
        raise ValueError("segments must be >= 2.")
    stroke = _bezier_points(
        (p0_u, p0_v),
        (p1_u, p1_v),
        (p2_u, p2_v),
        (p3_u, p3_v),
        segments,
    )
    return {"status": "success", "stroke": stroke, "strokes": [stroke]}


def transform_strokes(
    strokes_json: str,
    *,
    translate_u: float = 0.0,
    translate_v: float = 0.0,
    scale_u: float = 1.0,
    scale_v: float = 1.0,
    rotate_deg: float = 0.0,
    pivot_u: float = 0.5,
    pivot_v: float = 0.5,
) -> dict[str, object]:
    """Transform strokes by scale/rotate/translate around a pivot."""
    strokes = _coerce_strokes(strokes_json)
    angle = math.radians(rotate_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    transformed: list[list[tuple[float, float]]] = []
    for stroke in strokes:
        points: list[tuple[float, float]] = []
        for u, v in stroke:
            du = (u - pivot_u) * scale_u
            dv = (v - pivot_v) * scale_v
            ru = du * cos_a - dv * sin_a
            rv = du * sin_a + dv * cos_a
            points.append((ru + pivot_u + translate_u, rv + pivot_v + translate_v))
        transformed.append(points)

    return {"status": "success", "strokes": transformed}


def concat_strokes(strokes_json: str) -> dict[str, object]:
    """Combine multiple stroke lists into a single list of strokes."""
    payload = strokes_json
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("strokes_json must be valid JSON.") from exc

    if not isinstance(payload, list):
        raise ValueError("strokes_json must be a list.")

    def _is_point(item: object) -> bool:
        return (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        )

    def _is_stroke(item: object) -> bool:
        return isinstance(item, list) and (len(item) == 0 or all(_is_point(p) for p in item))

    strokes: list[list[tuple[float, float]]] = []
    if all(_is_stroke(item) for item in payload):
        strokes.extend(_coerce_strokes(payload))
    else:
        for entry in payload:
            strokes.extend(_coerce_strokes(entry))

    return {"status": "success", "strokes": strokes, "count": len(strokes)}
