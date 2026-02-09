from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()


import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pyniryo import NiryoRobot, PoseObject

from niryo_tools import close_robot, open_robot

DEFAULT_ROBOT_IP = "10.10.10.10"


def load_poses(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Pose file must be a JSON object mapping names to poses.")
    return data


def save_poses(path: Path, poses: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(poses, handle, indent=2, sort_keys=True)


def format_pose(pose: PoseObject) -> str:
    x, y, z, roll, pitch, yaw = pose.to_list()
    return (
        f"x={x:.4f} y={y:.4f} z={z:.4f} "
        f"roll={roll:.4f} pitch={pitch:.4f} yaw={yaw:.4f}"
    )


def confirm_overwrite(name: str) -> bool:
    answer = input(f"Pose '{name}' already exists. Overwrite? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def prompt_for_name() -> str:
    while True:
        name = input("Pose name (blank to skip): ").strip()
        if name:
            return name
        return ""


def wait_for_trigger_windows(robot: NiryoRobot, interval_s: float) -> str:
    import msvcrt

    while True:
        pose = robot.get_pose()
        print(format_pose(pose), flush=True)
        time.sleep(interval_s)
        if msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in {"\r", "\n"}:
                return "capture"
            if char.lower() == "q":
                return "quit"


def wait_for_trigger_threaded(robot: NiryoRobot, interval_s: float) -> str:
    result: dict[str, str] = {}

    def listener() -> None:
        line = input("Press Enter to capture, or type q then Enter to quit: ")
        result["action"] = "quit" if line.strip().lower() == "q" else "capture"

    thread = threading.Thread(target=listener, daemon=True)
    thread.start()

    while thread.is_alive():
        pose = robot.get_pose()
        print(format_pose(pose), flush=True)
        time.sleep(interval_s)

    return result.get("action", "capture")


def wait_for_trigger(robot: NiryoRobot, interval_s: float) -> str:
    if sys.platform == "win32":
        return wait_for_trigger_windows(robot, interval_s)
    return wait_for_trigger_threaded(robot, interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teach named poses and store them in JSON.")
    parser.add_argument("--ip", default=None, help="Robot IP address (overrides NIRYO_ROBOT_IP).")
    parser.add_argument(
        "--poses-file",
        default="data/poses.json",
        help="Path to the JSON file used to store poses.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between pose prints while waiting for input.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from the robot API.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable auto-calibration on connect.",
    )
    return parser.parse_args()


def resolve_robot_ip(ip_override: str | None) -> str:
    target_ip = ip_override or os.environ.get("NIRYO_ROBOT_IP", DEFAULT_ROBOT_IP)
    if not target_ip:
        raise ValueError("NIRYO_ROBOT_IP is required.")
    return target_ip


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")

    poses_path = Path(args.poses_file)
    poses = load_poses(poses_path)

    robot_ip = resolve_robot_ip(args.ip)
    robot = open_robot(
        robot_ip,
        verbose=args.verbose,
        auto_calibrate=not args.no_calibrate,
        ensure_learning_mode_off=False,
    )

    try:
        robot.set_learning_mode(True)
        print("Learning mode ON. Move the arm by hand.")
        print("Press Enter to capture, 'q' to quit.")

        while True:
            action = wait_for_trigger(robot, args.interval)
            if action == "quit":
                print("Exiting teaching loop.")
                break

            pose = robot.get_pose()
            name = prompt_for_name()
            if not name:
                print("Skipped pose.")
                continue
            if name in poses and not confirm_overwrite(name):
                print("Keeping existing pose.")
                continue

            poses[name] = pose.to_dict()
            save_poses(poses_path, poses)
            print(f"Saved '{name}' to {poses_path}.")
            print("Press Enter to capture, 'q' to quit.")
    finally:
        try:
            robot.set_learning_mode(False)
        finally:
            close_robot(robot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
