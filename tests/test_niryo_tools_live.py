from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from niryo_tools import (
    DEFAULT_MOVE_STEP_M,
    get_joints,
    get_pose,
    move_down,
    move_up,
    robot_session,
    set_arm_speed,
)


def _result_dir(test_name: str) -> Path:
    path = Path("test_results") / test_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.live_arm
def test_read_pose_and_joints() -> None:
    result_dir = _result_dir("test_read_pose_and_joints")
    with robot_session() as robot:
        set_arm_speed(robot, 30)
        pose = get_pose(robot)
        joints = get_joints(robot)

    (result_dir / "state.json").write_text(
        json.dumps({"pose": pose, "joints": joints}, indent=2),
        encoding="utf-8",
    )

    assert len(pose) == 6
    assert len(joints) == 6


@pytest.mark.live_arm
def test_move_up_down() -> None:
    result_dir = _result_dir("test_move_up_down")
    distance = DEFAULT_MOVE_STEP_M

    with robot_session() as robot:
        set_arm_speed(robot, 30)
        pose_before = get_pose(robot)
        move_up(robot, distance)
        pose_up = get_pose(robot)
        move_down(robot, distance)
        pose_after = get_pose(robot)

    (result_dir / "poses.json").write_text(
        json.dumps(
            {
                "pose_before": pose_before,
                "pose_up": pose_up,
                "pose_after": pose_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    z_before = pose_before[2]
    z_up = pose_up[2]
    z_after = pose_after[2]

    assert z_up > z_before + 0.005
    assert abs(z_after - z_before) < 0.01
