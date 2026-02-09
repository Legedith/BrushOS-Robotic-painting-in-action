from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from pyniryo import PoseObject

from niryo_tools import (
    close_robot,
    move_pose,
    open_robot,
    set_arm_speed,
)

DEFAULT_ROBOT_IP = "10.10.10.10"
DEFAULT_SEQUENCE = (
    "top_left",
    "bottom_right",
    "top_right",
    "bottom_left",
    "bowl_top",
    "bowl_bottom_1",
    "bowl_bottom_2",
)


def resolve_robot_ip(ip_override: str | None) -> str:
    target_ip = ip_override or os.environ.get("NIRYO_ROBOT_IP", DEFAULT_ROBOT_IP)
    if not target_ip:
        raise ValueError("NIRYO_ROBOT_IP is required.")
    return target_ip


def load_poses(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("poses file must be a JSON object mapping names to poses.")
    return payload


def pose_to_list(pose_dict: dict[str, Any]) -> list[float]:
    pose = PoseObject.from_dict(pose_dict)
    return pose.to_list()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark pose sequence.")
    parser.add_argument("--ip", default=None, help="Robot IP (overrides NIRYO_ROBOT_IP).")
    parser.add_argument("--poses-file", default="poses.json", help="Path to poses JSON.")
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause between moves.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help="Arm speed percentage (1-100).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable auto-calibration on connect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sequence without moving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pause < 0:
        raise ValueError("--pause must be >= 0.")
    if args.speed is not None and not (1 <= args.speed <= 100):
        raise ValueError("--speed must be between 1 and 100.")

    poses_path = Path(args.poses_file)
    poses = load_poses(poses_path)

    missing = [name for name in DEFAULT_SEQUENCE if name not in poses]
    if missing:
        raise ValueError(f"Missing poses in {poses_path}: {', '.join(missing)}")

    robot_ip = resolve_robot_ip(args.ip)
    robot = open_robot(
        robot_ip,
        verbose=False,
        auto_calibrate=not args.no_calibrate,
        ensure_learning_mode_off=True,
    )

    try:
        if args.speed is not None:
            set_arm_speed(robot, args.speed)

        print("Benchmark sequence:")
        for name in DEFAULT_SEQUENCE:
            print(f"  - {name}")
        print("")

        for name in DEFAULT_SEQUENCE:
            pose_list = pose_to_list(poses[name])
            print(f"Moving to {name} ...")
            if not args.dry_run:
                move_pose(robot, pose_list)
            if args.pause:
                time.sleep(args.pause)
    finally:
        close_robot(robot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
