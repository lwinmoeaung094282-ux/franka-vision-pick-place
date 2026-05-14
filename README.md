# FrankaVisionPick

A Franka Panda pick-and-place pipeline in **Isaac Sim 5.1** with semantic segmentation vision, natural language command parsing via LLMs, and a physics-aware stacking mode with automatic fall detection and re-stacking.

![Isaac Sim 5.1](https://img.shields.io/badge/Isaac%20Sim-5.1-76b900?logo=nvidia)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Platform](https://img.shields.io/badge/Platform-Windows%20%2B%20WSL2-0078D4?logo=windows)
![LLM](https://img.shields.io/badge/LLM-Anthropic%20%7C%20Qwen3-orange)

---

## What it does

- Spawns **4 objects** of different sizes and shapes (cube, tall pillar, flat tile, block) in Bin A
- The Franka Panda arm **detects each object** using a replicator semantic segmentation camera
- A custom **7-phase state machine** (hover → descend → grasp → lift → transport → lower → release) drives the arm via RMPFlow
- **Natural language instructions** parsed by an LLM tell the robot which objects to pick, in what order, and whether to skip any
- **Stacking mode** stacks all objects on top of each other with automatic fall detection, repair, and safe retreat between placements
- Runs with or without vision (`--no-vision` falls back to ground-truth positions)

---

## Architecture

```
run.py
├── build_scene()            table + 2 bins + Franka + 4 DynamicCuboid objects
├── VisionDetector           (vision.py)
│   ├── setup()              replicator camera + semantic segmentation annotators
│   └── detect(label)        seg mask → 2D centroid → ray-plane intersection → 3D XY
├── run_pick_place()         7-phase state machine using RMPFlowController
├── retreat_arm()            lifts gripper to safe height after each placement
├── check_and_repair_stack() verifies stack integrity; re-picks and re-stacks fallen blocks
└── language.py              LLM command parser (Anthropic API or Qwen3 via LM Studio)
```

### Vision pipeline

1. Replicator creates a pinhole camera with explicit `focal_length` so physical FOV matches the ray-cast math
2. Semantic segmentation mask isolates pixels for the target label (each object has a unique label)
3. 2D centroid of the mask is computed in image space
4. A ray is cast from the camera eye through that pixel
5. Ray intersects the horizontal plane at `z = TABLE_H + object_height / 2` (object centre)
6. Resulting 3D position is handed to the pick-and-place controller

### Language conditioning

The LLM is called **once before the simulation starts** and returns a pick order list (e.g. `[2, 3, 0, 1]`). After that, the LLM has no further involvement — RMPFlow and the state machine handle all robot motion.

```
User instruction → LLM → [2, 3, 0, 1]
                             ↓
                    RMPFlow executes each pick-place in that order
```

Two backends are supported and tried in order:

| Backend | How | Cost |
|---------|-----|------|
| **Qwen3-27B via LM Studio** | Local inference, OpenAI-compatible API at `localhost:1234` | Free |
| **Anthropic Claude Haiku** | Cloud API, pass `--api-key` | ~$0.001/call |

**What worked in testing:** Anthropic Claude Haiku reliably returned valid JSON and chose the correct stacking order (`[2, 3, 0, 1]` — widest base first) on the first call. Qwen3-27B via LM Studio connected and reasoned correctly in its chain-of-thought, but in thinking mode it leaves the `content` field empty and puts output in `reasoning_content` — the code extracts the order from there. When extraction fails, it falls back to Anthropic, then to a stable hardcoded default.

### Stacking mode

In `--stack` mode the robot places all objects on top of each other in Bin B rather than in separate grid slots:

- The LLM orders objects by stability (widest/flattest base first)
- After each placement the arm **retreats straight up** to HOVER_Z before moving back to Bin A, preventing gripper collisions with the placed stack
- Before picking each new block, `check_and_repair_stack()` verifies all previously placed blocks are still on the stack; any that fell are re-picked and re-stacked before continuing
- `stack_z` is **measured from actual object positions** each time, not a cumulative counter — so the robot always knows the true height of the stack

---

## Objects

| # | Shape | Size | Base Area | Color |
|---|-------|------|-----------|-------|
| 0 | Square cube | 4 × 4 × 4 cm | 16 cm² | Red |
| 1 | Tall pillar | 3 × 3 × 7 cm | 9 cm² | Blue |
| 2 | Wide flat tile | 5 × 5 × 3.5 cm | 25 cm² | Green |
| 3 | Medium block | 4 × 4 × 5.5 cm | 16 cm² | Yellow |

All horizontal dimensions ≤ 5 cm to fit within the 10 cm Franka gripper opening.
Stable stack order (widest base first): **2 → 3 → 0 → 1**

---

## Requirements

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1.0 (Windows) |
| Python | 3.11 (Isaac Sim embedded) |
| Conda env | `env_isaaclab` with `torch 2.7.0+cu128` |
| GPU | RTX-class (RaytracedLighting renderer) |
| OS | Windows 11 + WSL2 |
| LM Studio | Optional — for free local Qwen3 inference |
| Anthropic API | Optional — Claude Haiku for reliable cloud parsing |

---

## Setup

Isaac Sim's embedded Python lacks `torch`. Inject the `env_isaaclab` conda packages via `PYTHONPATH`:

```bat
set PYTHONPATH=C:\Users\<user>\miniconda3\envs\env_isaaclab\Lib\site-packages
```

Install the OpenAI client (needed for LM Studio) into env_isaaclab:

```bat
C:\Users\<user>\miniconda3\envs\env_isaaclab\python.exe -m pip install openai
```

---

## Running

All commands are run in Windows CMD.

### Grid placement (default) — transfer all 4 objects to individual slots in Bin B

```bat
set PYTHONPATH=C:\Users\<user>\miniconda3\envs\env_isaaclab\Lib\site-packages
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py
```

### With a natural language command (Anthropic)

```bat
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py ^
  --command "pick the blue pillar first, skip the yellow block" ^
  --api-key sk-ant-YOUR_KEY_HERE
```

### With a natural language command (LM Studio / Qwen3)

Start the LM Studio local server first (load Qwen3, click Start Server), then:

```bat
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py ^
  --command "pick the blue pillar first, skip the yellow block"
```

### Stacking mode — LLM decides order, stacks all blocks

```bat
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py ^
  --stack --api-key sk-ant-YOUR_KEY_HERE
```

### Skip vision (fastest, uses ground-truth positions)

```bat
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py --no-vision
```

### Headless

```bat
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py --headless --no-vision
```

### Multiple episodes

```bat
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\<user>\isaac_pick_place\run.py --episodes 3
```

---

## Project structure

```
├── run.py              Main script — scene, state machine, stacking logic, episode loop
├── vision.py           Camera detection — replicator, seg mask, ray-plane intersection
├── language.py         LLM command parser — Qwen3 (LM Studio) + Anthropic fallback
├── manual_control.py   Interactive keyboard control for debugging
├── launch.bat          Windows launcher
└── launch_manual.bat   Windows launcher for manual control
```

---

## Key implementation notes

- **`SimulationApp` must be instantiated before any other `omni.*` / `isaacsim.*` import** — violating this crashes the process silently
- RMPFlow targets the `right_gripper` frame (fingertip midpoint), not `panda_hand` — `GRASP_Z = TABLE_H + object_height / 2` aligns the fingertips with the object equator
- Bin walls are `VisualCuboid` (no physics collision) so the gripper can descend freely; the table surface acts as the bin floor
- Vision FOV is set explicitly via `focal_length` on the replicator camera so it matches the manual ray-cast math — mismatched FOV caused a systematic ~7 cm lateral offset in early testing
- Gripper orientation quaternion `[0, 0, 1, 0]` (wxyz) = 180° around Y axis = correct Franka top-down grasp. `[0, 1, 0, 0]` (180° around X) causes the gripper to push objects sideways
- **Passing API keys:** `set ANTHROPIC_API_KEY=...` does not reliably propagate through Isaac Sim's `python.bat`. Always pass secrets via `--api-key` on the command line
- **Qwen3 thinking mode:** Qwen3 models via LM Studio put chain-of-thought in `reasoning_content` and leave `content` empty. The parser uses `model_dump()` to access `reasoning_content` and extracts the order list with a regex
- **Stack repair:** `stack_z` is re-measured from actual object positions before each placement, not accumulated — so a knocked-over block is detected and re-stacked automatically

---

## Extending

| Goal | What to change |
|------|----------------|
| Add more objects | Extend `OBJ_DEFS` and `GRID_OFFSETS` in `run.py` and `OBJ_DESCRIPTIONS` in `language.py` |
| Different shapes | Swap `DynamicCuboid` for `DynamicSphere` / `DynamicCylinder` / any USD asset |
| YOLOv8 detection | Replace `VisionDetector.detect()` — same `(x, y, z)` return interface |
| Domain randomisation | Add `rep.randomizer.*` calls before `world.reset()` |
| RL training | Wrap `run_pick_place()` in a `gymnasium.Env`, connect to Isaac Lab |
| Different LLM | Add a new `_call_*` function in `language.py` following the same pattern |
