# Gemini 3: Eyes + Hands in the Real World

This project is **not about the Niryo arm**. The arm is just the body.  
The real question we asked was: **If Gemini 3 gets eyes and hands, how well can it understand its surroundings and act autonomously?**

We built a closed-loop system where Gemini 3 sees the scene, plans strokes, moves a real robot arm, checks the result, and improves. We then pushed it into art and produced **the first Gemini 3-powered real-world drawings**: not pixels, but ink on paper.

## Evidence (Images + Logs)

Real captures and transcript snippets:

![Drawing tools visible](docs/evidence/drawing_tools_check.jpg)
Tools check: the camera verified the brush in the gripper, paint bowl, and paper.

![Cave art v1](docs/evidence/final_cave_art_v1.jpg)
Cave art v1: the agent drew a bison and two hunters from strokes it generated.

![Cave art v2](docs/evidence/final_cave_art_v2.jpg)
Cave art v2: the agent added a sun and ground line to improve composition.

![Cave art v3](docs/evidence/final_cave_art_v3.jpg)
Cave art v3: final pass with birds + mountains for scene depth.

Run transcript: `docs/logs/run.logs`

```text
Tool call: build_bezier_stroke {"p0_u": 0.35, "p0_v": 0.45, ...}
Tool call: concat_strokes {"strokes_json": "[[[0.7,0.45],[0.75,0.55],[0.7,0.6]], ...]"}
Tool call: arm_draw_strokes {"strokes_json": "[[[0.35,0.45],...]]", "refill_after": 2}
Tool call: capture_photo_tool {"prefix": "final_cave_art_result", "output_dir": "captures\\agent_runs\\20260209_183722_draw_square_in_air"}
Tool output: arm_draw_strokes {"status": "success", "strokes": 8, "poses_file": "data/poses.json"}
```

## Why This Matters

This is not a text-to-image demo. Gemini 3 is controlling **physical action** in the real world. The output is constrained by friction, paint viscosity, gravity, and mechanical tolerances. It is real-world AI making real-world marks.

## Project Summary

**Goal:** Evaluate Gemini 3’s multimodal reasoning when it has **eyes (camera)** and **hands (robot arm)**, and see how far it can go without hardcoded drawings.

**Result:** Gemini 3 can:
- Observe the workspace and find the brush, bowl, and paper.
- Generate large-stroke geometry from scratch.
- Execute those strokes with refill logic.
- Review its own output and add corrective or compositional elements.

## Gemini Integration (200-Word Summary)

We use Gemini 3 as the central planner in a closed-loop visual manipulation system. The model receives live camera frames and generates structured tool calls rather than freeform text. It identifies objects in the scene, reasons about spatial layout, and decides when to move the arm, refill paint, or draw strokes. Gemini 3’s multimodal reasoning is used in three ways: (1) scene understanding to locate the brush, bowl, and paper; (2) compositional planning to decide what to draw and where; (3) post-action evaluation by comparing new captures to earlier ones and deciding whether to add or correct strokes. 

The drawing pipeline is intentionally constrained to large, continuous strokes to match a real brush. Gemini 3 generates strokes via geometry primitives (circles, arcs, S-curves, Beziers), transforms them, concatenates them, and executes them through the robot arm. The agent then captures fresh images and uses vision feedback to refine the composition. This showcases Gemini 3’s ability to connect perception, planning, and motor control in a physical environment, which goes far beyond a chat interface and highlights true multimodal autonomy.

## Technical Journey (Short)

1. **Pose teaching**  
   Built a learning-mode tool to capture paper corners + bowl poses into `data/poses.json`.

2. **Drawing primitives**  
   Implemented normalized stroke execution + refill logic for a thick brush.

3. **Autonomous agent**  
   Gave Gemini 3 building blocks to invent shapes and execute them without presets.

4. **Closed-loop refinement**  
   The agent captured images after major strokes and improved the scene iteratively.

## Challenges and Fixes

- Thick brush meant thin details disappeared. We enforced large strokes and frequent refills.
- Pose accuracy dominated quality. We standardized a 4-corner paper frame.
- Visual feedback was essential. The agent now captures and evaluates after major actions.
- The robot server warned about version mismatch; we pinned `pyniryo` 1.2.3 and documented upgrades.

## Architecture (High Level)

Camera → Gemini 3 reasoning → tool calls → robot execution → camera feedback → refinement loop

## Repo Layout

- `camera_agent/`: Gemini 3 agent + drawing tools
- `scripts/`: runnable entrypoints
- `data/poses.json`: taught poses
- `docs/evidence/`: curated images
- `docs/logs/run.logs`: run transcript

## How to Run

```bash
uv sync
uv run scripts/teach_poses.py --ip 10.10.10.10 --poses-file data/poses.json
uv run scripts/run_camera_agent_interactive.py
```

Example prompt:

```
Design a large cave-art scene with big strokes only.
Refill after every 1-2 strokes. Capture after each major addition.
```

## Hackathon Submission Checklist

- Gemini Integration summary: see "Gemini Integration" section above.
- Public project link: TODO
- Public code repository: this repo
- 3-minute demo video: TODO

## How We Meet Judging Criteria

Technical Execution: full closed-loop system, real robot control, vision feedback, robust tools.  
Potential Impact: demonstrates real-world autonomy with multimodal reasoning and physical actuation.  
Innovation / Wow: first Gemini 3-driven physical art pipeline with self-correction.  
Presentation / Demo: evidence images, run logs, and clear instructions included.

## Environment Variables

- `GOOGLE_API_KEY` or `GOOGLE_API_KEYS`: required for Gemini
- `NIRYO_ROBOT_IP`: optional override for robot IP (default `10.10.10.10`)
