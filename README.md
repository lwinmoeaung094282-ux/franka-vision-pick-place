# FrankaVisionPick

A Franka Panda pick-and-place pipeline in **Isaac Sim 5.1** that transfers four objects with different shapes from one bin to another using camera-based semantic segmentation for object detection.

![Isaac Sim 5.1](https://img.shields.io/badge/Isaac%20Sim-5.1-76b900?logo=nvidia)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Platform](https://img.shields.io/badge/Platform-Windows%20%2B%20WSL2-0078D4?logo=windows)

---

## What it does

- Spawns **4 objects** of different sizes and shapes (cube, tall pillar, flat tile, block) in Bin A
- The Franka Panda arm **detects each object** using a replicator semantic segmentation camera
- A custom **7-phase state machine** (hover → descend → grasp → lift → transport → lower → release) drives the arm via RMPFlow
- All 4 objects are transferred to matching slots in Bin B, one by one
- Runs with or without vision (`--no-vision` falls back to ground-truth positions)

---

## Architecture

```
run.py
├── build_scene()          table + 2 bins + Franka + 4 DynamicCuboid objects
├── VisionDetector         (vision.py)
│   ├── setup()            replicator camera + RGB + semantic segmentation annotators
│   └── detect(label)      seg mask → 2D centroid → ray-plane intersection → 3D XY
└── run_pick_place()       7-phase state machine using RMPFlowController
```

### Vision pipeline

1. Replicator creates a pinhole camera with explicit `focal_length` so physical FOV matches the ray-cast math
2. Semantic segmentation mask isolates pixels for the target label (each object has a unique label)
3. 2D centroid of the mask is computed in image space
4. A ray is cast from the camera eye through that pixel
5. Ray intersects the horizontal plane at `z = TABLE_H + object_height / 2` (object centre)
6. Resulting 3D position is handed to the pick-and-place controller

---

## Objects

| # | Shape | Size | Color |
|---|-------|------|-------|
| 0 | Square cube | 4 × 4 × 4 cm | Red |
| 1 | Tall pillar | 3 × 3 × 7 cm | Blue |
| 2 | Wide flat tile | 5 × 5 × 3.5 cm | Green |
| 3 | Medium block | 4 × 4 × 5.5 cm | Yellow |

All horizontal dimensions ≤ 5 cm to fit within the 10 cm Franka gripper opening.

---

## Requirements

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1.0 (Windows) |
| Python | 3.11 (Isaac Sim embedded) |
| Conda env | `env_isaaclab` with `torch 2.7.0+cu128` |
| GPU | RTX-class (RaytracedLighting renderer) |
| OS | Windows 11 + WSL2 |

---

## Setup

Isaac Sim's embedded Python lacks `torch`. Inject the `env_isaaclab` conda packages via `PYTHONPATH`:

```bat
set PYTHONPATH=<conda_env>\Lib\site-packages
```

Each `.bat` launcher already does this — just double-click or run from CMD.

---

## Running

### GUI mode (default)
```bat
launch.bat
```

### Skip vision (fastest, uses ground-truth positions)
```bat
launch.bat --no-vision
```

### Headless
```bat
launch.bat --headless --no-vision
```

### Multiple episodes
```bat
launch.bat --episodes 3 --no-vision
```

### Manual control (keyboard)
```bat
launch_manual.bat
```
Click inside the Isaac Sim viewport first, then:

| Key | Action |
|-----|--------|
| W / S | EEF +X / −X |
| A / D | EEF +Y / −Y |
| Q / E | EEF +Z / −Z |
| C | Close gripper |
| O | Open gripper |
| P | Print state |
| R | Reset target |
| ESC | Quit |

---

## Project structure

```
├── run.py              Main script — scene, state machine, episode loop
├── vision.py           Camera detection — replicator, seg mask, ray-plane
├── manual_control.py   Interactive keyboard control for debugging
├── launch.bat          Windows launcher for run.py
└── launch_manual.bat   Windows launcher for manual_control.py
```

---

## Key implementation notes

- **`SimulationApp` must be instantiated before any other `omni.*` / `isaacsim.*` import** — violating this crashes the process silently
- RMPFlow targets the `right_gripper` frame (fingertip midpoint), not `panda_hand` — `GRASP_Z = TABLE_H + object_height / 2` aligns the fingertips with the object equator
- Bin walls are `VisualCuboid` (no physics collision) so the gripper can descend freely; the table surface acts as the bin floor
- Vision FOV is set explicitly via `focal_length` on the replicator camera so it matches the manual ray-cast math — mismatched FOV was the root cause of a systematic ~7 cm lateral offset in early testing
- Gripper orientation quaternion `[0, 0, 1, 0]` (wxyz) = 180° around Y axis = correct Franka top-down grasp. `[0, 1, 0, 0]` (180° around X) is wrong and causes the gripper to push objects sideways

---

## Extending

| Goal | What to change |
|------|----------------|
| Add more objects | Extend `OBJ_DEFS` and `GRID_OFFSETS` |
| Different shapes | Swap `DynamicCuboid` for `DynamicSphere` / `DynamicCylinder` / any USD asset |
| YOLOv8 detection | Replace `VisionDetector.detect()` — same `(x, y, z)` return interface |
| Domain randomisation | Add `rep.randomizer.*` calls before `world.reset()` |
| RL training | Wrap `run_pick_place()` in a `gymnasium.Env`, connect to Isaac Lab |
