from __future__ import annotations

import os
from typing import Callable, TypeVar

from google.adk.agents.llm_agent import Agent
from pyniryo import NiryoRobot

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_ROBOT_IP = "10.10.10.10"
DEFAULT_STEP_M = 0.05

T = TypeVar("T")


def _robot_ip(ip: str | None) -> str:
    """Resolve the robot IP address."""
    target_ip = ip or os.environ.get("NIRYO_ROBOT_IP", DEFAULT_ROBOT_IP)
    if not target_ip:
        raise ValueError("NIRYO_ROBOT_IP is required.")
    return target_ip


def _with_robot(ip: str | None, action: Callable[[NiryoRobot], T]) -> T:
    """Connect to the robot, run an action, and close."""
    robot_ip = _robot_ip(ip)
    robot = NiryoRobot(robot_ip)
    robot.connect()
    try:
        return action(robot)
    finally:
        robot.close_connection()


def _prepare_motion(robot: NiryoRobot) -> None:
    """Disable learning mode before motion."""
    robot.set_learning_mode(False)


def _validate_step(step_m: float) -> None:
    """Validate a step in meters."""
    if step_m <= 0:
        raise ValueError("step_m must be > 0.")


def get_pose(ip: str | None = None) -> dict[str, object]:
    """Get the current robot pose."""

    def _action(robot: NiryoRobot) -> dict[str, object]:
        pose = robot.get_pose()
        return {"pose": pose.to_dict()}

    return _with_robot(ip, _action) | {"status": "success"}


def get_joints(ip: str | None = None) -> dict[str, object]:
    """Get the current joint positions."""

    def _action(robot: NiryoRobot) -> dict[str, object]:
        joints = robot.get_joints()
        return {"joints": joints.to_list()}

    return _with_robot(ip, _action) | {"status": "success"}


def calibrate_auto(ip: str | None = None) -> dict[str, object]:
    """Auto-calibrate the robot."""

    def _action(robot: NiryoRobot) -> dict[str, object]:
        robot.calibrate_auto()
        return {}

    return _with_robot(ip, _action) | {"status": "success"}


def go_home(ip: str | None = None) -> dict[str, object]:
    """Move the robot to the home pose."""

    def _action(robot: NiryoRobot) -> dict[str, object]:
        _prepare_motion(robot)
        robot.move_to_home_pose()
        pose = robot.get_pose()
        return {"pose": pose.to_dict()}

    return _with_robot(ip, _action) | {"status": "success"}


def move_relative(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    ip: str | None = None,
) -> dict[str, object]:
    """Move the robot relative (meters, radians)."""

    def _action(robot: NiryoRobot) -> dict[str, object]:
        _prepare_motion(robot)
        robot.move_relative(x, y, z, roll, pitch, yaw)
        pose = robot.get_pose()
        return {"pose": pose.to_dict()}

    return _with_robot(ip, _action) | {"status": "success"}


def move_up(step_m: float = DEFAULT_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot up in Z."""
    _validate_step(step_m)
    return move_relative(z=step_m, ip=ip)


def move_down(step_m: float = DEFAULT_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot down in Z."""
    _validate_step(step_m)
    return move_relative(z=-step_m, ip=ip)


def move_left(step_m: float = DEFAULT_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot left in Y."""
    _validate_step(step_m)
    return move_relative(y=step_m, ip=ip)


def move_right(step_m: float = DEFAULT_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot right in Y."""
    _validate_step(step_m)
    return move_relative(y=-step_m, ip=ip)


def move_forward(step_m: float = DEFAULT_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot forward in X."""
    _validate_step(step_m)
    return move_relative(x=step_m, ip=ip)


def move_backward(step_m: float = DEFAULT_STEP_M, ip: str | None = None) -> dict[str, object]:
    """Move the robot backward in X."""
    _validate_step(step_m)
    return move_relative(x=-step_m, ip=ip)


root_agent = Agent(
    name="niryo_arm_agent",
    model=DEFAULT_MODEL,
    description="Agent that controls a Niryo Ned robot arm.",
    instruction=(
        "You control a Niryo Ned arm. "
        "Use move_up/down/left/right/forward/backward for simple motion. "
        "Use move_relative for precise moves. "
        "Use get_pose/get_joints to inspect state. "
        "Use go_home or calibrate_auto when needed."
    ),
    tools=[
        get_pose,
        get_joints,
        calibrate_auto,
        go_home,
        move_relative,
        move_up,
        move_down,
        move_left,
        move_right,
        move_forward,
        move_backward,
    ],
)
