"""Shared scene configuration for OpenArm lift-style tasks.

This scene reproduces the official Isaac-Lift-Cube-OpenArm-Play-v0 layout:

  - OpenArm robot fixed at the standard base position (env origin, floor level).
  - Ground plane only – **no table or workbench**.
  - DexCube object(s) placed at standard lift-task positions (z ≈ 0.055 m).
  - EE frame transformer from openarm_link0 to openarm_ee_tcp.
  - Dome light.

For cube stacking, five DexCubes replace the single lift-task cube.  They are
placed in a fan pattern in front of the robot, all matching the reachable
workspace of the floor-mounted arm.

Key values (from the official OpenArm lift config)
---------------------------------------------------
  robot         : OPENARM_UNI_CFG (prim_path = {ENV_REGEX_NS}/Robot)
  arm joints    : openarm_joint.*
  gripper joints: openarm_finger_joint.*   open=0.044, close=0.0
  EE body       : openarm_hand
  EE TCP frame  : openarm_ee_tcp
  EE root link  : openarm_link0
  cube z (floor): 0.055 m  (centre of DexCube at scale 0.8 resting on floor)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

# ─────────────────────────────────────────────────────────────────────────────
# Cube / stacking constants
# ─────────────────────────────────────────────────────────────────────────────

# DexCube edge length in metres (scale 0.8 applied to the USD asset).
# This value is also the vertical step between stacked-cube centres:
#   target_z_i = CUBE_GROUND_Z + i * CUBE_HEIGHT
CUBE_HEIGHT: float = 0.055

# Z coordinate of a DexCube's centre when it rests on the floor.
# Matches the official OpenArm lift-task spawn position.
CUBE_GROUND_Z: float = 0.055

# "Planner floor z": the effective ground level seen by ClassicalStackPlanner.
# Derived so that the planner formula
#   table_top_z + cube_size / 2 + i * cube_size
# equals CUBE_GROUND_Z + i * CUBE_HEIGHT.
PLANNER_FLOOR_Z: float = CUBE_GROUND_Z - CUBE_HEIGHT / 2.0  # = 0.0275

# Aliases used by downstream modules and the planner config.
CUBE_SIZE: float = CUBE_HEIGHT       # alias – same value
TABLE_TOP_Z: float = PLANNER_FLOOR_Z  # alias for planner compatibility

NUM_CUBES: int = 5
CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{i}" for i in range(NUM_CUBES))

# ─────────────────────────────────────────────────────────────────────────────
# Default spawn positions  (local env frame, env origins added at runtime)
# ─────────────────────────────────────────────────────────────────────────────

LIFT_STYLE_CUBE_SPAWN_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.35, -0.16, CUBE_GROUND_Z),
    (0.35, -0.08, CUBE_GROUND_Z),
    (0.40,  0.00, CUBE_GROUND_Z),
    (0.35,  0.08, CUBE_GROUND_Z),
    (0.35,  0.16, CUBE_GROUND_Z),
)

# Stack base: (x, y, z) of the *first* (bottom) cube's centre in the stack.
LIFT_STYLE_STACK_BASE_LOCAL_POS: tuple[float, float, float] = (0.55, 0.0, CUBE_GROUND_Z)

# ─────────────────────────────────────────────────────────────────────────────
# EE debug markers
# ─────────────────────────────────────────────────────────────────────────────

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)
_EE_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"

_EE_TARGET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_TARGET_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
_EE_TARGET_MARKER_CFG.prim_path = "/Visuals/EETarget"

# ─────────────────────────────────────────────────────────────────────────────
# DexCube factory
# ─────────────────────────────────────────────────────────────────────────────


def make_dex_cube_cfg(
    name: str,
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Return a :class:`RigidObjectCfg` for a single DexCube placed at *pos*.

    Uses the Isaac Nucleus DexCube asset at scale 0.8, matching the official
    OpenArm lift environment.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Base scene  (robot + ground + light + EE frame, no cubes)
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class OpenArmLiftStyleSceneCfg(InteractiveSceneCfg):
    """OpenArm lift-style scene: robot at floor level, no table.

    Exactly matches the visual layout of Isaac-Lift-Cube-OpenArm-Play-v0:

    - OpenArm at env origin (not raised to any table height).
    - Ground plane.
    - Dome light.
    - EE frame transformer: ``openarm_link0`` → ``openarm_ee_tcp``.

    Sub-classes add cube objects for stacking tasks.
    """

    robot: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # EE frame: source = robot base link, target = TCP site
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=_EE_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full scene with five DexCubes
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class OpenArmLiftStyleWithCubesSceneCfg(OpenArmLiftStyleSceneCfg):
    """Lift-style scene with five DexCubes for sequential stacking.

    Cube positions mirror the official OpenArm lift spawn area, spread into a
    fan pattern so every cube is reachable from the floor-mounted arm:

      cube_0: (0.35, -0.16, 0.055)
      cube_1: (0.35, -0.08, 0.055)
      cube_2: (0.40,  0.00, 0.055)
      cube_3: (0.35,  0.08, 0.055)
      cube_4: (0.35,  0.16, 0.055)

    Stack base: (0.55, 0.0, 0.055)
    """

    cube_0 = make_dex_cube_cfg("cube_0", LIFT_STYLE_CUBE_SPAWN_POSITIONS[0])
    cube_1 = make_dex_cube_cfg("cube_1", LIFT_STYLE_CUBE_SPAWN_POSITIONS[1])
    cube_2 = make_dex_cube_cfg("cube_2", LIFT_STYLE_CUBE_SPAWN_POSITIONS[2])
    cube_3 = make_dex_cube_cfg("cube_3", LIFT_STYLE_CUBE_SPAWN_POSITIONS[3])
    cube_4 = make_dex_cube_cfg("cube_4", LIFT_STYLE_CUBE_SPAWN_POSITIONS[4])
