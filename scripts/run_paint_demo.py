from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()


import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

from pyniryo import PoseObject

from niryo_tools import close_robot, move_pose, open_robot, set_arm_speed

DEFAULT_ROBOT_IP = "10.10.10.10"

PAPER_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
BOWL_POSES = ("bowl_top", "bowl_bottom_1", "bowl_bottom_2")


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


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_pose(p0: Iterable[float], p1: Iterable[float], t: float) -> list[float]:
    return [lerp(a, b, t) for a, b in zip(p0, p1)]


def bilinear_pose(
    top_left: list[float],
    top_right: list[float],
    bottom_left: list[float],
    bottom_right: list[float],
    u: float,
    v: float,
) -> list[float]:
    top = lerp_pose(top_left, top_right, u)
    bottom = lerp_pose(bottom_left, bottom_right, u)
    return lerp_pose(top, bottom, v)


def lift_pose(pose: list[float], lift_m: float) -> list[float]:
    lifted = list(pose)
    lifted[2] += lift_m
    return lifted


def move_and_pause(robot, pose: list[float], pause_s: float, dry_run: bool) -> None:
    if dry_run:
        return
    move_pose(robot, pose)
    if pause_s:
        time.sleep(pause_s)


def dip_paint(
    robot,
    bowl_top: list[float],
    bowl_bottom_1: list[float],
    bowl_bottom_2: list[float],
    pause_s: float,
    dry_run: bool,
) -> None:
    move_and_pause(robot, bowl_top, pause_s, dry_run)
    move_and_pause(robot, bowl_bottom_1, pause_s, dry_run)
    move_and_pause(robot, bowl_bottom_2, pause_s, dry_run)
    move_and_pause(robot, bowl_top, pause_s, dry_run)


def move_line(
    robot,
    start_pose: list[float],
    end_pose: list[float],
    steps: int,
    pause_s: float,
    dry_run: bool,
) -> None:
    if steps < 1:
        steps = 1
    for i in range(1, steps + 1):
        t = i / steps
        pose = lerp_pose(start_pose, end_pose, t)
        move_and_pause(robot, pose, pause_s, dry_run)


def map_local_to_paper(u: float, v: float, box: tuple[float, float, float, float]) -> tuple[float, float]:
    u0, v0, u1, v1 = box
    return (lerp(u0, u1, u), lerp(v0, v1, v))


def circle_points(segments: int) -> list[tuple[float, float]]:
    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        u = 0.5 + 0.45 * math.cos(angle)
        v = 0.5 + 0.45 * math.sin(angle)
        points.append((u, v))
    points.append(points[0])
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paint demo using named poses.")
    parser.add_argument("--ip", default=None, help="Robot IP (overrides NIRYO_ROBOT_IP).")
    parser.add_argument("--poses-file", default="data/poses.json", help="Path to poses JSON.")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.2,
        help="Seconds to pause between waypoint moves.",
    )
    parser.add_argument(
        "--lift",
        type=float,
        default=0.02,
        help="Lift distance in meters between strokes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Waypoints per line segment.",
    )
    parser.add_argument(
        "--refill-after",
        type=int,
        default=2,
        help="Refill after this many line segments on paper.",
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
    if args.lift < 0:
        raise ValueError("--lift must be >= 0.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.refill_after < 1:
        raise ValueError("--refill-after must be >= 1.")
    if args.speed is not None and not (1 <= args.speed <= 100):
        raise ValueError("--speed must be between 1 and 100.")

    poses_path = Path(args.poses_file)
    poses = load_poses(poses_path)

    missing = [name for name in (*PAPER_CORNERS, *BOWL_POSES) if name not in poses]
    if missing:
        raise ValueError(f"Missing poses in {poses_path}: {', '.join(missing)}")

    tl = pose_to_list(poses["top_left"])
    tr = pose_to_list(poses["top_right"])
    bl = pose_to_list(poses["bottom_left"])
    br = pose_to_list(poses["bottom_right"])

    bowl_top = pose_to_list(poses["bowl_top"])
    bowl_bottom_1 = pose_to_list(poses["bowl_bottom_1"])
    bowl_bottom_2 = pose_to_list(poses["bowl_bottom_2"])

    robot_ip = resolve_robot_ip(args.ip)
    robot = open_robot(
        robot_ip,
        verbose=False,
        auto_calibrate=not args.no_calibrate,
        ensure_learning_mode_off=True,
    )

    shapes: list[tuple[str, list[list[tuple[float, float]]], tuple[float, float, float, float]]] = [
        (
            "X",
            [
                [(0.0, 0.0), (1.0, 1.0)],
                [(1.0, 0.0), (0.0, 1.0)],
            ],
            (0.08, 0.08, 0.45, 0.45),
        ),
        (
            "O",
            [circle_points(12)],
            (0.55, 0.08, 0.92, 0.45),
        ),
        (
            "L",
            [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]],
            (0.08, 0.55, 0.45, 0.92),
        ),
        (
            "Triangle",
            [[(0.5, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.0)]],
            (0.55, 0.55, 0.92, 0.92),
        ),
    ]

    try:
        if args.speed is not None:
            set_arm_speed(robot, args.speed)

        segment_counter = 0
        print("Starting paint demo.")
        dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run)

        for shape_name, strokes, box in shapes:
            print(f"Drawing {shape_name} ...")
            for stroke in strokes:
                if len(stroke) < 2:
                    continue

                start_u, start_v = map_local_to_paper(stroke[0][0], stroke[0][1], box)
                start_pose = bilinear_pose(tl, tr, bl, br, start_u, start_v)

                move_and_pause(robot, lift_pose(start_pose, args.lift), args.pause, args.dry_run)
                move_and_pause(robot, start_pose, args.pause, args.dry_run)

                for (u0, v0), (u1, v1) in zip(stroke, stroke[1:]):
                    global_u0, global_v0 = map_local_to_paper(u0, v0, box)
                    global_u1, global_v1 = map_local_to_paper(u1, v1, box)
                    p0 = bilinear_pose(tl, tr, bl, br, global_u0, global_v0)
                    p1 = bilinear_pose(tl, tr, bl, br, global_u1, global_v1)
                    move_line(robot, p0, p1, args.steps, args.pause, args.dry_run)
                    segment_counter += 1

                    if segment_counter >= args.refill_after:
                        print("Refilling paint ...")
                        move_and_pause(robot, lift_pose(p1, args.lift), args.pause, args.dry_run)
                        dip_paint(
                            robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run
                        )
                        segment_counter = 0

                end_u, end_v = map_local_to_paper(stroke[-1][0], stroke[-1][1], box)
                end_pose = bilinear_pose(tl, tr, bl, br, end_u, end_v)
                move_and_pause(robot, lift_pose(end_pose, args.lift), args.pause, args.dry_run)

        print("Demo complete.")
    finally:
        close_robot(robot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
