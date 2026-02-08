from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

from pyniryo import CalibrateMode, NiryoRobot, RobotAxis
from pyniryo.api.enums_communication import Command

DEFAULT_NIRYO_IP = "10.10.10.10"
DEFAULT_MOVE_STEP_M = 0.02
DEFAULT_ROTATE_STEP_RAD = 0.1


def open_robot(
    ip_address: str = DEFAULT_NIRYO_IP,
    *,
    verbose: bool = True,
    auto_calibrate: bool = True,
    ensure_learning_mode_off: bool = True,
) -> NiryoRobot:
    """Open a robot connection."""
    robot = NiryoRobot(ip_address, verbose=verbose)
    if auto_calibrate and robot.need_calibration():
        robot.calibrate_auto()
    if ensure_learning_mode_off and robot.get_learning_mode():
        robot.set_learning_mode(False)
    return robot


def close_robot(robot: NiryoRobot) -> None:
    """Close the robot connection."""
    robot.close_connection()
def _send_command(robot: NiryoRobot, command: Command, *params: object) -> None:
    """Send a raw command to the robot."""
    sender = robot._NiryoRobot__send_n_receive
    sender(command, *params)


@contextmanager
def robot_session(
    ip_address: str = DEFAULT_NIRYO_IP,
    *,
    verbose: bool = True,
    auto_calibrate: bool = True,
    ensure_learning_mode_off: bool = True,
) -> Iterator[NiryoRobot]:
    """Context manager for opening and closing a robot session."""
    robot = open_robot(
        ip_address,
        verbose=verbose,
        auto_calibrate=auto_calibrate,
        ensure_learning_mode_off=ensure_learning_mode_off,
    )
    try:
        yield robot
    finally:
        close_robot(robot)


def set_arm_speed(robot: NiryoRobot, percentage_speed: int) -> None:
    """Limit arm max velocity to a percentage."""
    robot.set_arm_max_velocity(percentage_speed)


def calibrate_auto(robot: NiryoRobot) -> None:
    """Run automatic calibration if needed."""
    if robot.need_calibration():
        robot.calibrate_auto()


def calibrate_manual(robot: NiryoRobot) -> None:
    """Run manual calibration."""
    robot.calibrate(CalibrateMode.MANUAL)


def get_joints(robot: NiryoRobot) -> list[float]:
    """Get the current joints in radians."""
    joints = robot.get_joints()
    return list(joints)


def get_pose(robot: NiryoRobot) -> list[float]:
    """Get the current pose as [x, y, z, roll, pitch, yaw]."""
    pose = robot.get_pose()
    return pose.to_list()


def move_joints(robot: NiryoRobot, joints: Iterable[float]) -> None:
    """Move to joint positions in radians."""
    values = list(joints)
    if len(values) != 6:
        raise ValueError("Expected 6 joint values.")
    robot.move_joints(values)


def move_pose(robot: NiryoRobot, pose: Iterable[float]) -> None:
    """Move to an absolute pose."""
    values = list(pose)
    if len(values) != 6:
        raise ValueError("Expected 6 pose values.")
    _send_command(robot, Command.MOVE_POSE, *values)


def move_home(robot: NiryoRobot) -> None:
    """Move to home pose."""
    robot.move_to_home_pose()


def shift_axis(robot: NiryoRobot, axis: RobotAxis, shift_value: float) -> None:
    """Shift along a single axis."""
    _send_command(robot, Command.SHIFT_POSE, axis, shift_value)


def move_up(
    robot: NiryoRobot,
    distance_m: float = DEFAULT_MOVE_STEP_M,
) -> None:
    """Move the end effector up."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    shift_axis(robot, RobotAxis.Z, distance_m)


def move_down(
    robot: NiryoRobot,
    distance_m: float = DEFAULT_MOVE_STEP_M,
) -> None:
    """Move the end effector down."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    shift_axis(robot, RobotAxis.Z, -distance_m)


def move_forward(
    robot: NiryoRobot,
    distance_m: float = DEFAULT_MOVE_STEP_M,
) -> None:
    """Move the end effector forward."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    shift_axis(robot, RobotAxis.X, distance_m)


def move_backward(
    robot: NiryoRobot,
    distance_m: float = DEFAULT_MOVE_STEP_M,
) -> None:
    """Move the end effector backward."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    shift_axis(robot, RobotAxis.X, -distance_m)


def move_left(
    robot: NiryoRobot,
    distance_m: float = DEFAULT_MOVE_STEP_M,
) -> None:
    """Move the end effector left."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    shift_axis(robot, RobotAxis.Y, distance_m)


def move_right(
    robot: NiryoRobot,
    distance_m: float = DEFAULT_MOVE_STEP_M,
) -> None:
    """Move the end effector right."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    shift_axis(robot, RobotAxis.Y, -distance_m)


def rotate_yaw(
    robot: NiryoRobot,
    angle_rad: float = DEFAULT_ROTATE_STEP_RAD,
) -> None:
    """Rotate the end effector around yaw."""
    if angle_rad == 0:
        raise ValueError("angle_rad must be non-zero.")
    shift_axis(robot, RobotAxis.YAW, angle_rad)
