# =============================================================================
# Isaac Sim 5.1 — Franka Multi-Object Bin-to-Bin Pick-and-Place
#
# Run from Windows:
#   set PYTHONPATH=C:\Users\Owner\miniconda3\envs\env_isaaclab\Lib\site-packages
#   C:\isaacsim\python.bat  \\wsl$\Ubuntu\home\lwin\isaac_pick_place\run.py
#   C:\isaacsim\python.bat  \\wsl$\Ubuntu\home\lwin\isaac_pick_place\run.py --headless
#   C:\isaacsim\python.bat  \\wsl$\Ubuntu\home\lwin\isaac_pick_place\run.py --no-vision
#   C:\isaacsim\python.bat  \\wsl$\Ubuntu\home\lwin\isaac_pick_place\run.py --episodes 3
#
# IMPORTANT: SimulationApp must be instantiated before any other omni/isaacsim imports.
# =============================================================================

import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--headless",   action="store_true")
parser.add_argument("--no-vision",  action="store_true")
parser.add_argument("--episodes",   type=int, default=1)
parser.add_argument("--command",    type=str, default=None,
                    help='Natural language pick instruction, e.g. "pick the blue pillar first"')
parser.add_argument("--api-key",    type=str, default=None,
                    help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})

import numpy as np
import carb
from enum import Enum, auto

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_labels
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision    import VisionDetector
from language  import parse_command


# =============================================================================
# Scene constants
# =============================================================================

TABLE_H     = 0.40
TABLE_CTR_Z = TABLE_H / 2

BIN_INNER_W = 0.24           # wider to hold 2×2 grid of objects
BIN_WALL_T  = 0.012
BIN_WALL_H  = 0.06

BIN_A_CTR   = np.array([0.45,  0.24])   # pick bin  (+Y side)
BIN_B_CTR   = np.array([0.45, -0.24])   # place bin (-Y side)

HOVER_Z     = TABLE_H + 0.30            # 0.70 m — safe travel height

# 180° around Y axis — gripper pointing straight down
ORIENT_DOWN = np.array([0.0, 0.0, 1.0, 0.0])

# 2×2 placement grid inside each bin (7 cm between centres)
_G = 0.070 / 2
GRID_OFFSETS = np.array([
    [-_G, -_G],
    [+_G, -_G],
    [-_G, +_G],
    [+_G, +_G],
])

# Object definitions: (name, label, size_xyz_metres, color_rgb)
#   All horizontal dims ≤ 5 cm so the 10 cm gripper opening can straddle any of them.
OBJ_DEFS = [
    ("obj_0", "obj_0",
     np.array([0.040, 0.040, 0.040]), np.array([0.90, 0.15, 0.10])),  # red  square cube
    ("obj_1", "obj_1",
     np.array([0.030, 0.030, 0.070]), np.array([0.20, 0.40, 0.90])),  # blue tall pillar
    ("obj_2", "obj_2",
     np.array([0.050, 0.050, 0.035]), np.array([0.15, 0.80, 0.25])),  # green wide flat tile
    ("obj_3", "obj_3",
     np.array([0.040, 0.040, 0.055]), np.array([0.95, 0.80, 0.10])),  # yellow medium-tall block
]

# Camera (vision detection)
CAM_EYE    = (1.1, 0.0, 1.4)
CAM_TARGET = (0.45, 0.0, TABLE_H)


# =============================================================================
# State machine
# =============================================================================

class Phase(Enum):
    HOVER_PICK  = auto()
    DESCEND     = auto()
    GRASP       = auto()
    LIFT        = auto()
    TRANSPORT   = auto()
    LOWER_PLACE = auto()
    RELEASE     = auto()
    DONE        = auto()

PHASE_BUDGET = {
    Phase.HOVER_PICK:  200,
    Phase.DESCEND:     200,
    Phase.GRASP:       150,
    Phase.LIFT:        200,
    Phase.TRANSPORT:   300,
    Phase.LOWER_PLACE: 200,
    Phase.RELEASE:     100,
}

POS_TOL_MOVE  = 0.025
POS_TOL_GRASP = 0.012
HAND_OFFSET_Z = 0.10   # panda_hand is ~10 cm above the right_gripper target


# =============================================================================
# Scene helpers
# =============================================================================

def add_bin(world, name, center_xy, color):
    """Four visual walls sitting on the table — no floor panel."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    outer  = BIN_INNER_W + 2 * BIN_WALL_T
    z_wall = TABLE_H + BIN_WALL_H / 2
    for side, sign in (("front", +1), ("back", -1)):
        world.scene.add(VisualCuboid(
            name=f"{name}_{side}", prim_path=f"/World/{name}_{side}",
            position=np.array([cx, cy + sign * (BIN_INNER_W / 2 + BIN_WALL_T / 2), z_wall]),
            scale=np.array([outer, BIN_WALL_T, BIN_WALL_H]), size=1.0, color=color,
        ))
    for side, sign in (("right", +1), ("left", -1)):
        world.scene.add(VisualCuboid(
            name=f"{name}_{side}", prim_path=f"/World/{name}_{side}",
            position=np.array([cx + sign * (BIN_INNER_W / 2 + BIN_WALL_T / 2), cy, z_wall]),
            scale=np.array([BIN_WALL_T, BIN_INNER_W, BIN_WALL_H]), size=1.0, color=color,
        ))


def build_scene(assets_root):
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    world.scene.add(FixedCuboid(
        name="table", prim_path="/World/Table",
        position=np.array([0.45, 0.0, TABLE_CTR_Z]),
        scale=np.array([1.0, 0.8, TABLE_H]), size=1.0,
        color=np.array([0.55, 0.35, 0.15]),
    ))

    add_bin(world, "bin_a", BIN_A_CTR, np.array([0.85, 0.30, 0.10]))
    add_bin(world, "bin_b", BIN_B_CTR, np.array([0.15, 0.45, 0.80]))

    asset_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")

    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.01, 0.01]),
        action_deltas=np.array([0.01, 0.01]),
    )
    franka = world.scene.add(SingleManipulator(
        prim_path="/World/Franka", name="franka",
        end_effector_prim_path="/World/Franka/panda_hand",
        gripper=gripper,
        position=np.array([0.0, 0.0, TABLE_H]),
    ))
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)

    objs = []
    for i, (name, label, size, color) in enumerate(OBJ_DEFS):
        xy  = BIN_A_CTR + GRID_OFFSETS[i]
        obj = world.scene.add(DynamicCuboid(
            name=name, prim_path=f"/World/{name.capitalize()}",
            position=np.array([xy[0], xy[1], TABLE_H + size[2] / 2]),
            scale=size, size=1.0, color=color,
        ))
        add_labels(obj.prim, labels=[label], instance_name="class")
        objs.append(obj)

    return world, franka, objs


# =============================================================================
# Per-object state machine
# =============================================================================

def run_pick_place(world, franka, controller, art_ctrl,
                   hover_pick, grasp_tgt, hover_place, place_tgt,
                   tag, max_steps=3000):
    """
    Execute one pick-and-place cycle.
    Returns True when the object has been released at the place target,
    False on timeout.
    """

    def eef_close_to(target, tol):
        hand_pos, _ = franka.end_effector.get_world_pose()
        expected    = target.copy()
        expected[2] += HAND_OFFSET_Z
        return float(np.linalg.norm(hand_pos - expected)) < tol

    phase        = Phase.HOVER_PICK
    phase_step   = 0
    gripper_open = True
    target       = hover_pick

    for _ in range(max_steps):
        world.step(render=True)
        if not world.is_playing():
            continue

        phase_step += 1

        if phase == Phase.HOVER_PICK:
            target = hover_pick
            if eef_close_to(target, POS_TOL_MOVE) or phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.DESCEND; phase_step = 0
                print(f"    [{tag}] → DESCEND")

        elif phase == Phase.DESCEND:
            target = grasp_tgt
            if eef_close_to(target, POS_TOL_GRASP) or phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.GRASP; phase_step = 0; gripper_open = False
                print(f"    [{tag}] → GRASP  (closing gripper)")

        elif phase == Phase.GRASP:
            target = grasp_tgt
            if phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.LIFT; phase_step = 0
                print(f"    [{tag}] → LIFT")

        elif phase == Phase.LIFT:
            target = hover_pick
            if eef_close_to(target, POS_TOL_MOVE) or phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.TRANSPORT; phase_step = 0
                print(f"    [{tag}] → TRANSPORT")

        elif phase == Phase.TRANSPORT:
            target = hover_place
            if eef_close_to(target, POS_TOL_MOVE) or phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.LOWER_PLACE; phase_step = 0
                print(f"    [{tag}] → LOWER_PLACE")

        elif phase == Phase.LOWER_PLACE:
            target = place_tgt
            if eef_close_to(target, POS_TOL_GRASP) or phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.RELEASE; phase_step = 0; gripper_open = True
                print(f"    [{tag}] → RELEASE  (opening gripper)")

        elif phase == Phase.RELEASE:
            target = place_tgt
            if phase_step >= PHASE_BUDGET[phase]:
                phase = Phase.DONE

        elif phase == Phase.DONE:
            return True

        arm_actions = controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=ORIENT_DOWN,
        )
        art_ctrl.apply_action(arm_actions)

        gripper_action = franka.gripper.forward("open" if gripper_open else "close")
        art_ctrl.apply_action(gripper_action)

    return False   # timeout


# =============================================================================
# Main
# =============================================================================

def main():
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Nucleus not reachable.")
        simulation_app.close()
        sys.exit(1)

    detector = VisionDetector(
        cam_eye=CAM_EYE, cam_target=CAM_TARGET,
        fov_deg=60, resolution=(256, 256),
        label=OBJ_DEFS[0][1],
    )

    # ---- Language command ---------------------------------------------------
    if args.command:
        instruction = args.command
    else:
        print("\nObjects available:")
        print("  0 — red square cube     (4×4×4 cm)")
        print("  1 — blue tall pillar    (3×3×7 cm)")
        print("  2 — green flat tile     (5×5×3.5 cm)")
        print("  3 — yellow medium block (4×4×5.5 cm)")
        instruction = input("\nInstruction (Enter to pick all in default order): ").strip()

    if instruction:
        print(f"\n[LANG] Parsing: \"{instruction}\"")
        pick_order = parse_command(instruction, api_key=args.api_key)
    else:
        pick_order = [0, 1, 2, 3]

    print(f"[LANG] Pick order: {pick_order}  "
          f"({', '.join(OBJ_DEFS[i][0] for i in pick_order)})")

    all_results = []

    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{args.episodes}  —  {len(pick_order)} objects, Bin A → Bin B")
        print(f"{'='*60}")

        world, franka, objs = build_scene(assets_root)
        world.reset()

        if not args.no_vision:
            detector.setup()

        controller = RMPFlowController(name="rmpflow_ctrl", robot_articulation=franka)
        art_ctrl   = franka.get_articulation_controller()

        ep_ok = []

        for i in pick_order:
            obj                   = objs[i]
            name, label, size, _  = OBJ_DEFS[i]
            grasp_z  = TABLE_H + size[2] / 2
            pick_xy  = BIN_A_CTR + GRID_OFFSETS[i]
            place_xy = BIN_B_CTR + GRID_OFFSETS[i]

            print(f"\n  ── Object {i+1}/4  [{name}]  "
                  f"size={np.round(size*100,1)} cm  grasp_z={grasp_z:.3f} m")

            # ---- Vision / ground-truth detection ----------------------------
            if args.no_vision:
                detected = obj.get_local_pose()[0]
                print(f"    [VISION] skipped — gt {np.round(detected, 3)}")
            else:
                print(f"    [VISION] detecting '{label}' …")
                detected = detector.detect(table_z=grasp_z, label=label)
                if detected is not None:
                    gt  = obj.get_local_pose()[0]
                    err = np.round(detected[:2] - gt[:2], 3)
                    print(f"    [VISION] detected={np.round(detected,3)}  "
                          f"gt={np.round(gt,3)}  xy_err={err}")
                else:
                    detected = obj.get_local_pose()[0]
                    print(f"    [VISION] failed → fallback gt {np.round(detected,3)}")

            # ---- Compute waypoints ------------------------------------------
            px, py = float(detected[0]),  float(detected[1])
            bx, by = float(place_xy[0]),  float(place_xy[1])

            hover_pick  = np.array([px, py, HOVER_Z])
            grasp_tgt   = np.array([px, py, grasp_z])
            hover_place = np.array([bx, by, HOVER_Z])
            place_tgt   = np.array([bx, by, grasp_z])

            print(f"    pick  hover={np.round(hover_pick,3)}  "
                  f"grasp={np.round(grasp_tgt,3)}")
            print(f"    place hover={np.round(hover_place,3)}  "
                  f"target={np.round(place_tgt,3)}")

            # ---- Execute ----------------------------------------------------
            ok = run_pick_place(world, franka, controller, art_ctrl,
                                hover_pick, grasp_tgt, hover_place, place_tgt,
                                name)

            if ok:
                final = obj.get_local_pose()[0]
                dist  = float(np.linalg.norm(final[:2] - place_xy))
                ok    = dist < 0.12
                print(f"    [RESULT] final={np.round(final,3)}  "
                      f"dist_to_slot={dist:.3f} m  {'OK' if ok else 'FAIL'}")
            else:
                print(f"    [RESULT] TIMEOUT")

            ep_ok.append(ok)

        all_results.append(ep_ok)

    simulation_app.close()

    print(f"\n{'='*60}")
    for ep, res in enumerate(all_results):
        n = sum(res)
        print(f"Episode {ep+1}: {n}/{len(res)} objects transferred")
        for idx, (obj_i, ok) in enumerate(zip(pick_order, res)):
            print(f"  {OBJ_DEFS[obj_i][0]:6s} ({np.round(OBJ_DEFS[obj_i][2]*100,0)} cm): "
                  f"{'OK' if ok else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
