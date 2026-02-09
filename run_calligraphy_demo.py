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

HAIKU_LINES = ("SOFT WIND", "STILL WATER", "MOON DRAWS RING")

LINE_SPACING = 0.55
LETTER_SPACING = 0.25
WORD_SPACING = 0.6


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
) -> int:
    if len(points) < 2:
        return 0

    start_pose = bilinear_pose(tl, tr, bl, br, points[0][0], points[0][1])
    move_and_pause(robot, lift_pose(start_pose, lift_m), pause_s, dry_run)
    move_and_pause(robot, start_pose, pause_s, dry_run)

    segments = 0
    for (u0, v0), (u1, v1) in zip(points, points[1:]):
        p0 = bilinear_pose(tl, tr, bl, br, u0, v0)
        p1 = bilinear_pose(tl, tr, bl, br, u1, v1)
        move_line(robot, p0, p1, steps, pause_s, dry_run)
        segments += 1

    end_pose = bilinear_pose(tl, tr, bl, br, points[-1][0], points[-1][1])
    move_and_pause(robot, lift_pose(end_pose, lift_m), pause_s, dry_run)
    return segments


def circle_points(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        t = i / segments
        angle = lerp(start_angle, end_angle, t)
        u = center[0] + radius * math.cos(angle)
        v = center[1] + radius * math.sin(angle)
        points.append((u, v))
    return points


def bezier_points(
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


LETTER_STROKES: dict[str, list[list[tuple[float, float]]]] = {
    "A": [
        [(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)],
        [(0.2, 0.6), (0.8, 0.6)],
    ],
    "D": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(0.0, 0.0), (0.75, 0.2), (0.75, 0.8), (0.0, 1.0)],
    ],
    "E": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(0.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.5), (0.7, 0.5)],
        [(0.0, 1.0), (1.0, 1.0)],
    ],
    "F": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(0.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.5), (0.7, 0.5)],
    ],
    "G": [
        [(1.0, 0.2), (0.8, 0.0), (0.2, 0.0), (0.0, 0.2), (0.0, 0.8), (0.2, 1.0), (0.8, 1.0), (1.0, 0.8), (0.6, 0.8)],
    ],
    "I": [
        [(0.5, 0.0), (0.5, 1.0)],
    ],
    "L": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(0.0, 1.0), (1.0, 1.0)],
    ],
    "M": [
        [(0.0, 1.0), (0.0, 0.0)],
        [(0.0, 0.0), (0.5, 0.6), (1.0, 0.0)],
        [(1.0, 0.0), (1.0, 1.0)],
    ],
    "N": [
        [(0.0, 1.0), (0.0, 0.0)],
        [(0.0, 0.0), (1.0, 1.0)],
        [(1.0, 1.0), (1.0, 0.0)],
    ],
    "O": [
        circle_points((0.5, 0.5), 0.5, 0.0, 2 * math.pi, 12),
    ],
    "R": [
        [(0.0, 1.0), (0.0, 0.0)],
        [(0.0, 0.0), (0.7, 0.0), (0.7, 0.5), (0.0, 0.5)],
        [(0.0, 0.5), (1.0, 1.0)],
    ],
    "S": [
        [(1.0, 0.0), (0.2, 0.0), (0.0, 0.2), (0.0, 0.5), (0.8, 0.5), (1.0, 0.7), (1.0, 1.0), (0.2, 1.0)],
    ],
    "T": [
        [(0.0, 0.0), (1.0, 0.0)],
        [(0.5, 0.0), (0.5, 1.0)],
    ],
    "W": [
        [(0.0, 0.0), (0.25, 1.0), (0.5, 0.4), (0.75, 1.0), (1.0, 0.0)],
    ],
}


def measure_line(text: str) -> float:
    width = 0.0
    for char in text:
        if char == " ":
            width += WORD_SPACING
        else:
            width += 1.0 + LETTER_SPACING
    if width:
        width -= LETTER_SPACING
    return width


def layout_text_block(
    lines: Iterable[str],
    box: tuple[float, float, float, float],
) -> list[list[tuple[float, float]]]:
    u0, v0, u1, v1 = box
    width_box = u1 - u0
    height_box = v1 - v0

    lines = [line.upper() for line in lines]
    line_widths = [measure_line(line) for line in lines]
    max_width = max(line_widths) if line_widths else 1.0
    total_height_units = len(lines) * 1.0 + (len(lines) - 1) * LINE_SPACING
    scale = min(width_box / max_width, height_box / total_height_units)

    y_start = v0 + (height_box - total_height_units * scale) / 2
    strokes: list[list[tuple[float, float]]] = []

    for row, line in enumerate(lines):
        line_width = line_widths[row]
        x_start = u0 + (width_box - line_width * scale) / 2
        cursor_x = x_start
        cursor_y = y_start + row * (1.0 + LINE_SPACING) * scale

        for char in line:
            if char == " ":
                cursor_x += WORD_SPACING * scale
                continue
            glyph = LETTER_STROKES.get(char)
            if not glyph:
                cursor_x += (1.0 + LETTER_SPACING) * scale
                continue
            for stroke in glyph:
                transformed = [
                    (cursor_x + x * scale, cursor_y + y * scale) for x, y in stroke
                ]
                strokes.append(transformed)
            cursor_x += (1.0 + LETTER_SPACING) * scale
    return strokes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calligraphy + haiku demo.")
    parser.add_argument("--ip", default=None, help="Robot IP (overrides NIRYO_ROBOT_IP).")
    parser.add_argument("--poses-file", default="poses.json", help="Path to poses JSON.")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.15,
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
        default=3,
        help="Waypoints per line segment.",
    )
    parser.add_argument(
        "--refill-after",
        type=int,
        default=2,
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

    try:
        if args.speed is not None:
            set_arm_speed(robot, args.speed)

        stroke_count = 0
        print("Starting calligraphy demo.")
        dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run)

        # Enso circle (two passes for richer ink).
        center = (0.35, 0.33)
        radius = 0.23
        gap = math.radians(25)
        enso_points = circle_points(center, radius, gap, 2 * math.pi - gap, 48)
        enso_points_inner = circle_points(center, radius - 0.015, gap, 2 * math.pi - gap, 48)
        for points in (enso_points, enso_points_inner):
            stroke_count += 1
            stroke_polyline(
                robot,
                points,
                tl=tl,
                tr=tr,
                bl=bl,
                br=br,
                lift_m=args.lift,
                steps=args.steps,
                pause_s=args.pause,
                dry_run=args.dry_run,
            )
            if stroke_count >= args.refill_after:
                print("Refilling paint ...")
                dip_paint(
                    robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run
                )
                stroke_count = 0

        # Sweeping stroke beneath the enso.
        sweep = bezier_points(
            (0.18, 0.60),
            (0.32, 0.45),
            (0.58, 0.72),
            (0.82, 0.55),
            32,
        )
        stroke_count += 1
        stroke_polyline(
            robot,
            sweep,
            tl=tl,
            tr=tr,
            bl=bl,
            br=br,
            lift_m=args.lift,
            steps=args.steps,
            pause_s=args.pause,
            dry_run=args.dry_run,
        )
        if stroke_count >= args.refill_after:
            print("Refilling paint ...")
            dip_paint(robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run)
            stroke_count = 0

        # Haiku text block on the right.
        text_box = (0.56, 0.62, 0.95, 0.93)
        text_strokes = layout_text_block(HAIKU_LINES, text_box)
        for stroke in text_strokes:
            stroke_count += 1
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
            if stroke_count >= args.refill_after:
                print("Refilling paint ...")
                dip_paint(
                    robot, bowl_top, bowl_bottom_1, bowl_bottom_2, args.pause, args.dry_run
                )
                stroke_count = 0

        print("Calligraphy demo complete.")
    finally:
        close_robot(robot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
