# Real-World Brush Drawing with a Niryo Arm

This project turns a Niryo Ned robot arm + camera into a physical drawing system. The robot learns a paper frame, refills a brush from a bowl, and executes big, bold strokes on paper from natural-language instructions. The result is real, physical ink on paper, not a synthetic render.

We call the current capability "caveman-level autonomy": the robot can draw recognizable large-stroke scenes on command, refine them using camera feedback, and keep itself fed with paint.

## Evidence (Images + Logs)

Below are real captures from the system while the agent was running.

![Drawing tools visible](docs/evidence/drawing_tools_check.jpg)
Tools check: the camera verified the brush in the gripper, paint bowl, and paper.

![Cave art v1](docs/evidence/final_cave_art_v1.jpg)
Cave art v1: the agent drew a bison and two hunters from strokes it generated.

![Cave art v2](docs/evidence/final_cave_art_v2.jpg)
Cave art v2: the agent added a sun and ground line to improve composition.

![Cave art v3](docs/evidence/final_cave_art_v3.jpg)
Cave art v3: final pass with birds + mountains for scene depth.

Run transcript snippets are in `docs/logs/run.logs`. Example excerpt showing the agent generating strokes and executing them:

```text
Tool call: build_bezier_stroke {"p0_u": 0.35, "p0_v": 0.45, ...}
Tool call: concat_strokes {"strokes_json": "[[[0.7,0.45],[0.75,0.55],[0.7,0.6]], ...]"}
Tool call: arm_draw_strokes {"strokes_json": "[[[0.35,0.45],...]]", "refill_after": 2}
Tool call: capture_photo_tool {"prefix": "final_cave_art_result", "output_dir": "captures\\agent_runs\\20260209_183722_draw_square_in_air"}
Tool output: arm_draw_strokes {"status": "success", "strokes": 8, "poses_file": "data/poses.json"}
```

## System Overview

Core idea: represent all drawing strokes as normalized 2D points on the paper (u=0..1, v=0..1). Convert those to real robot poses using four taught corner poses. Then move the arm along those curves while refilling paint on a schedule.

Key components:
- `scripts/teach_poses.py`: teaches named poses (paper corners + bowl) into `data/poses.json`.
- `camera_agent/arm_tools.py`: drawing + refill primitives.
- `camera_agent/agent.py`: the interactive agent with vision + motion + drawing guidance.
- `scripts/run_camera_agent_interactive.py`: interactive REPL to ask the agent to draw.
- `docs/logs/run.logs`: a real run with prompts + outputs.

## What We Built (Technical Journey)

1. Pose teaching
   We built a learning-mode tool to capture named poses into `data/poses.json`. These include `top_left`, `top_right`, `bottom_left`, `bottom_right`, `bowl_top`, `bowl_bottom_1`, `bowl_bottom_2`.

2. Drawing primitives
   Implemented large-stroke motion from normalized 2D points and added paint-refill logic (`bowl_top -> bowl_bottom_1 -> bowl_bottom_2 -> bowl_top`).

3. Agent autonomy
   Equipped the agent with drawing building blocks (`build_circle_stroke`, `build_arc_stroke`, `build_s_curve_stroke`, `build_bezier_stroke`, `transform_strokes`, `concat_strokes`, `arm_draw_strokes`, `arm_refill_paint`). The agent now synthesizes its own shapes and draws them without presets.

4. Closed-loop refinement
   The agent captures images before and after drawing, evaluates them, and adds strokes to improve composition.

## Challenges and Fixes

- Thick brush behavior meant thin strokes disappeared. We constrained the agent to large, continuous strokes and frequent refills.
- Pose accuracy mattered more than expected. We added a dedicated pose-teaching workflow and re-used those four paper corners everywhere.
- Visual feedback was essential. The agent now captures images after major strokes to decide whether to add or correct.
- The robot server warned about version mismatch in logs; we kept the pyniryo API pinned to 1.2.3 and documented the requirement to update robot firmware when possible.

## How to Run

### 1) Install dependencies

```bash
uv sync
```

Requires Python 3.13+ (see `pyproject.toml`).

### 2) Teach poses (one-time per setup)

```bash
uv run scripts/teach_poses.py --ip 10.10.10.10 --poses-file data/poses.json
```

Move the arm by hand to each target, press Enter, and name the pose.

### 3) Run the interactive agent

```bash
uv run scripts/run_camera_agent_interactive.py
```

Example prompt:

```
Design a large cave-art scene with big strokes only. Use the bowl refill between strokes.
Capture after each major addition and refine the composition.
```

## Building Blocks for Drawing (No Presets)

The agent now constructs drawings using these primitives.

Geometry strokes:
- `build_circle_stroke(center_u, center_v, radius, segments)`
- `build_arc_stroke(center_u, center_v, radius, start_deg, end_deg, segments)`
- `build_s_curve_stroke(center_u, center_v, radius, segments)`
- `build_bezier_stroke(p0_u, p0_v, p1_u, p1_v, p2_u, p2_v, p3_u, p3_v, segments)`

Stroke operations:
- `transform_strokes(strokes_json, translate_u, translate_v, scale_u, scale_v, rotate_deg, pivot_u, pivot_v)`
- `concat_strokes(strokes_json)`

Execution:
- `arm_refill_paint(poses_file, pause_s, speed)`
- `arm_draw_strokes(strokes_json, poses_file, refill_after, lift_m, steps)`

## Repository Layout (Relevant)

- `camera_agent/agent.py`: interactive agent instructions + tools.
- `camera_agent/arm_tools.py`: drawing primitives and robot execution.
- `scripts/teach_poses.py`: pose capture workflow.
- `data/poses.json`: the taught poses for this setup (example).
- `docs/logs/run.logs`: the session transcript.
- `docs/evidence/*.jpg`: captured evidence images.

## Demo Scripts

- `scripts/run_pose_benchmark.py`: sanity check for paper corners and bowl moves.
- `scripts/run_paint_demo.py`: draws simple shapes with periodic refills.
- `scripts/run_calligraphy_demo.py`: enso + haiku layout (scripted).
- `scripts/run_big_poster_calligraphy.py`: vertical poster-scale strokes.
- `scripts/run_yinyang_poster.py`: yin-yang poster (scripted).

## Environment Variables

- `GOOGLE_API_KEY` or `GOOGLE_API_KEYS`: required for Gemini.
- `NIRYO_ROBOT_IP`: optional override for the robot IP (defaults to `10.10.10.10`).

## Known Constraints

- Thick brushes demand large strokes; tiny details are intentionally skipped.
- Accuracy depends on pose teaching and camera alignment.
- Paint refilling is mandatory for consistent line density.

## Why This Matters

This is not a text-to-image demo. The system generates and executes physical actions in the real world. The resulting art is constrained by gravity, friction, paint viscosity, and hardware tolerances, making the output more meaningful than pure pixels. It is real-world AI making real-world marks.
