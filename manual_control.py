# =============================================================================
# Isaac Sim 5.1 — Manual Franka control for grasp debugging
#
# Run from Windows (GUI mode only):
#   set PYTHONPATH=C:\Users\Owner\miniconda3\envs\env_isaaclab\Lib\site-packages
#   C:\isaacsim\python.bat  \\wsl$\Ubuntu\home\lwin\isaac_pick_place\manual_control.py
#
# Keys (click inside the Isaac Sim VIEWPORT to capture keyboard input):
#   W / S     — EEF +X / -X  (forward / back)
#   A / D     — EEF +Y / -Y  (left / right)
#   Q / E     — EEF +Z / -Z  (up / down)
#   C         — close gripper
#   O         — open gripper
#   P         — print current state
#   R         — reset EEF target above Bin A
#   ESC       — quit
# =============================================================================

import sys
import os

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import numpy as np
import carb
import carb.input as carb_input
import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_labels
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Scene constants (mirror run.py)
# =============================================================================

CUBE_SIZE   = 0.040
CUBE_HALF   = CUBE_SIZE / 2
TABLE_H     = 0.40
TABLE_CTR_Z = TABLE_H / 2
BIN_INNER_W = 0.16
BIN_WALL_T  = 0.012
BIN_WALL_H  = 0.04
BIN_A_CTR   = np.array([0.45,  0.22])
BIN_B_CTR   = np.array([0.45, -0.22])
CUBE_Z      = TABLE_H + CUBE_HALF       # 0.426 m
LABEL       = "pick_cube"

STEP_XY     = 0.005   # metres per physics step while key held
STEP_Z      = 0.005

START_TARGET = np.array([BIN_A_CTR[0], BIN_A_CTR[1], TABLE_H + 0.35])

# Orientation: gripper pointing straight down (180 deg around X, wxyz)
ORIENT_DOWN  = np.array([0.0, 0.0, 1.0, 0.0])


# =============================================================================
# Keyboard state (filled by carb callback, read in main loop)
# =============================================================================

keys_held     = set()   # keys currently pressed/repeated
cmd_close     = False
cmd_open      = False
cmd_print     = False
cmd_reset     = False
cmd_quit      = False

KI = carb_input.KeyboardInput   # shorthand

MOVE_KEYS = {
    KI.W, KI.S, KI.A, KI.D, KI.Q, KI.E,
}


def on_keyboard_event(event, *args, **kwargs):
    global cmd_close, cmd_open, cmd_print, cmd_reset, cmd_quit

    k = event.input
    t = event.type

    if t == carb_input.KeyboardEventType.KEY_PRESS:
        keys_held.add(k)
        if k == KI.C:       cmd_close = True
        elif k == KI.O:     cmd_open  = True
        elif k == KI.P:     cmd_print = True
        elif k == KI.R:     cmd_reset = True
        elif k == KI.ESCAPE: cmd_quit = True

    elif t == carb_input.KeyboardEventType.KEY_REPEAT:
        keys_held.add(k)   # keep held during auto-repeat

    elif t == carb_input.KeyboardEventType.KEY_RELEASE:
        keys_held.discard(k)

    return True   # don't consume the event


# =============================================================================
# Scene helpers
# =============================================================================

def add_bin(world, name, center_xy, color):
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

    cube = world.scene.add(DynamicCuboid(
        name="pick_cube", prim_path="/World/PickCube",
        position=np.array([BIN_A_CTR[0], BIN_A_CTR[1], CUBE_Z]),
        scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]), size=1.0,
        color=np.array([0.9, 0.15, 0.10]),
    ))
    add_labels(cube.prim, labels=[LABEL], instance_name="class")
    return world, franka, cube


# =============================================================================
# State printer
# =============================================================================

def print_state(franka, cube, target_pos, gripper_open):
    eef_pos, _     = franka.end_effector.get_world_pose()
    cube_pos, _    = cube.get_local_pose()
    all_joints     = franka.get_joint_positions()
    gripper_joints = all_joints[-2:]
    dist           = float(np.linalg.norm(eef_pos - cube_pos))

    print("\n┌─── State ────────────────────────────────────────")
    print(f"│  Target pos      : {np.round(target_pos, 4)}")
    print(f"│  EEF pos         : {np.round(eef_pos, 4)}")
    print(f"│  Cube pos        : {np.round(cube_pos, 4)}")
    print(f"│  EEF → cube dist : {dist:.4f} m")
    print(f"│  EEF z - cube z  : {eef_pos[2] - cube_pos[2]:+.4f} m")
    print(f"│  Finger joints   : {np.round(gripper_joints, 4)}"
          f"  ({'OPEN' if gripper_open else 'CLOSED'})")
    print("└──────────────────────────────────────────────────")


# =============================================================================
# Main
# =============================================================================

def main():
    global cmd_close, cmd_open, cmd_print, cmd_reset, cmd_quit

    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Nucleus not reachable.")
        simulation_app.close()
        sys.exit(1)

    world, franka, cube = build_scene(assets_root)
    world.reset()

    # --- Wire up keyboard via carb.input ------------------------------------
    input_iface  = carb_input.acquire_input_interface()
    app_window   = omni.appwindow.get_default_app_window()
    keyboard     = app_window.get_keyboard()
    kb_sub       = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    controller   = RMPFlowController(name="rmpflow_ctrl", robot_articulation=franka)
    art_ctrl     = franka.get_articulation_controller()
    target_pos   = START_TARGET.copy()
    gripper_open = True
    step_count   = 0

    print("\n" + "="*54)
    print("  MANUAL FRANKA CONTROL  —  Isaac Sim 5.1")
    print("="*54)
    print("  IMPORTANT: click inside the Isaac Sim VIEWPORT first")
    print("  W/S  → EEF ±X    A/D  → EEF ±Y    Q/E  → EEF ±Z")
    print("  C    → close gripper    O  → open gripper")
    print("  P    → print state      R  → reset target")
    print("  ESC  → quit")
    print(f"\n  Cube is at : [{BIN_A_CTR[0]:.3f}, {BIN_A_CTR[1]:.3f}, {CUBE_Z:.3f}]")
    print(f"  Start target: {np.round(target_pos, 3)}")
    print("="*54 + "\n")

    while simulation_app.is_running():
        world.step(render=True)

        if not world.is_playing():
            continue

        step_count += 1

        # --- One-shot commands -----------------------------------------------
        if cmd_quit:
            break

        if cmd_close:
            gripper_open = False
            cmd_close    = False
            print("  [GRIPPER] CLOSE")

        if cmd_open:
            gripper_open = True
            cmd_open     = False
            print("  [GRIPPER] OPEN")

        if cmd_print:
            print_state(franka, cube, target_pos, gripper_open)
            cmd_print = False

        if cmd_reset:
            target_pos = START_TARGET.copy()
            cmd_reset  = False
            print(f"  [RESET] target → {np.round(target_pos, 3)}")

        # --- Continuous movement from held keys ------------------------------
        if KI.W in keys_held:  target_pos[0] += STEP_XY
        if KI.S in keys_held:  target_pos[0] -= STEP_XY
        if KI.A in keys_held:  target_pos[1] += STEP_XY
        if KI.D in keys_held:  target_pos[1] -= STEP_XY
        if KI.Q in keys_held:  target_pos[2] += STEP_Z
        if KI.E in keys_held:  target_pos[2] -= STEP_Z

        # --- Arm (RMPFlow) ---------------------------------------------------
        arm_actions = controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=ORIENT_DOWN,
        )
        art_ctrl.apply_action(arm_actions)

        # --- Gripper ---------------------------------------------------------
        gripper_action = franka.gripper.forward("open" if gripper_open else "close")
        art_ctrl.apply_action(gripper_action)

        # --- Periodic state print --------------------------------------------
        if step_count % 300 == 0:
            eef_pos, _ = franka.end_effector.get_world_pose()
            cube_pos, _ = cube.get_local_pose()
            dist = float(np.linalg.norm(eef_pos - cube_pos))
            fingers = franka.get_joint_positions()[-2:]
            print(f"  [step {step_count:5d}]  EEF={np.round(eef_pos,3)}"
                  f"  cube={np.round(cube_pos,3)}"
                  f"  dist={dist:.3f}m"
                  f"  fingers={np.round(fingers,3)}"
                  f"  gripper={'open' if gripper_open else 'closed'}")

    input_iface.unsubscribe_to_keyboard_events(keyboard, kb_sub)
    simulation_app.close()


if __name__ == "__main__":
    main()
