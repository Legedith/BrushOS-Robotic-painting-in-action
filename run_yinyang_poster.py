from __future__ import annotations

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


def stroke_polyline(
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
    dry_run: bool,
) -> None:
    if len(points) < 2:
        return

    start_pose = bilinear_pose(tl, tr, bl, br, points[0][0], points[0][1])
    move_and_pause(robot, lift_pose(start_pose, lift_m), pause_s, dry_run)
    move_and_pause(robot, start_pose, pause_s, dry_run)

    for (u0, v0), (u1, v1) in zip(points, points[1:]):
        p0 = bilinear_pose(tl, tr, bl, br, u0, v0)
        p1 = bilinear_pose(tl, tr, bl, br, u1, v1)
        move_line(robot, p0, p1, steps, pause_s, dry_run)

    end_pose = bilinear_pose(tl, tr, bl, br, points[-1][0], points[-1][1])
    move_and_pause(robot, lift_pose(end_pose, lift_m), pause_s, dry_run)


def circle_points(center: tuple[float, float], radius: float, segments: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        u = center[0] + radius * math.cos(angle)
        v = center[1] + radius * math.sin(angle)
        points.append((u, v))
    return points


def s_curve_points(
    center: tuple[float, float],
    radius: float,
    segments: int,
) -> list[tuple[float, float]]:
    cx, cy = center
    r = radius / 2
    points: list[tuple[float, float]] = []

    # Upper half: right-side arc from top to center.
    for i in range(segments + 1):
        t = i / segments
        angle = math.radians(90 - 180 * t)
        u = cx + r * math.cos(angle)
        v = (cy - r) + r * math.sin(angle)
        points.append((u, v))

    # Lower half: left-side arc from center to bottom.
    for i in range(segments + 1):
        t = i / segments
        angle = math.radians(90 + 180 * t)
        u = cx + r * math.cos(angle)
        v = (cy + r) + r * math.sin(angle)
        points.append((u, v))
    return points


KANJI_STROKES: dict[str, list[list[tuple[float, float]]]] = {
    "山": [
        [(0.2, 0.05), (0.2, 0.95)],
        [(0.5, 0.0), (0.5, 0.95)],
        [(0.8, 0.05), (0.8, 0.95)],
        [(0.2, 0.95), (0.8, 0.95)],
    ],
    "川": [
        [(0.2, 0.05), (0.2, 0.95)],
        [(0.5, 0.05), (0.5, 0.95)],
        [(0.8, 0.05), (0.8, 0.95)],
    ],
}


def layout_kanji(
    characters: list[str],
    boxes: list[tuple[float, float, float, float]],
) -> list[list[tuple[float, float]]]:
    strokes: list[list[tuple[float, float]]] = []
    for char, box in zip(characters, boxes):
        glyph = KANJI_STROKES.get(char)
        if not glyph:
            continue
        u0, v0, u1, v1 = box
        for stroke in glyph:
            transformed = [(lerp(u0, u1, x), lerp(v0, v1, y)) for x, y in stroke]
            strokes.append(transformed)
    return strokes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Big yin-yang poster with kanji strokes.")
    parser.add_argument("--ip", default=None, help="Robot IP (overrides NIRYO_ROBOT_IP).")
    parser.add_argument("--poses-file", default="poses.json", help="Path to poses JSON.")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.12,
        help="Seconds to pause between waypoint moves.",
    )
    parser.add_argument(
        "--lift",
        type=float,
        default=0.03,
        help="Lift distance in meters between strokes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Waypoints per line segment (small to keep strokes bold).",
    )
    parser.add_argument(
        "--refill-after",
        type=int,
        default=1,
        help="Refill after this many strokes on paper.",
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

    # Big taiji in the upper half.
    center = (0.5, 0.33)
    radius = 0.28
    strokes: list[list[tuple[float, float]]] = [
        circle_points(center, radius, 48),
        circle_points(center, radius - 0.015, 48),
        s_curve_points(center, radius, 24),
        s_curve_points(center, radius - 0.015, 24),
        circle_points((center[0], center[1] - radius / 2), radius * 0.18, 20),
        circle_points((center[0], center[1] + radius / 2), radius * 0.18, 20),
    ]

    # Two large kanji below: 山 (mountain) and 川 (river).
    strokes.extend(
        layout_kanji(
            ["山", "川"],
            [
                (0.18, 0.62, 0.82, 0.78),
                (0.18, 0.80, 0.82, 0.96),
            ],
        )
    )

    try:
        if args.speed is not None:
            set_arm_speed(robot, args.speed)

        print("Starting yin-yang poster.")
        dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run)

        stroke_count = 0
        for stroke in strokes:
            if stroke_count >= args.refill_after:
                print("Refilling paint ...")
                dip_paint(
                    robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run
                )
                stroke_count = 0

            stroke_polyline(
                robot,
                stroke,
                tl=tl,
                tr=tr,
                bl=bl,
                br=br,
                lift_m=args.lift,
                steps=args.steps,
                pause_s=args.pause,
                dry_run=args.dry_run,
            )
            stroke_count += 1

        print("Poster complete.")
    finally:
        close_robot(robot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
