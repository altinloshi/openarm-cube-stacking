"""Shared scene configuration that reproduces the official OpenArm lift setup.

This module is the single source of truth for the cube-stacking scene geometry.
It mirrors ``Isaac-Lift-Cube-OpenArm-Play-v0`` exactly so every Nepher
cube-stacking task shares the same visual layout as the upstream OpenArm lift
environment:

* The OpenArm is fixed at the standard lift base pose (root at the env origin,
  i.e. resting on the lab-table top surface at ``z = 0``).  The robot is **not**
  raised onto a custom workbench.
* A ``SeattleLabTable`` USD sits at ``[0.5, 0, 0]`` (rotated upright) and the
  ground plane is dropped to ``z = -1.05`` so the table top is flush with
  ``z = 0`` — identical to the upstream lift scene.
* The manipulated object is the NVIDIA ``DexCube`` (``scale = 0.8``) which the
  upstream lift task spawns at ``z = 0.055``.  For cube stacking the single
  lift object is replaced by five colour-coded DexCubes placed in the same
  reachable area in front of the arm.
* The end-effector frame transformer uses the official OpenArm frame names:
  root ``openarm_link0`` → target ``openarm_ee_tcp``.

Layout (local env frame; env origins are added at runtime):

         +x (forward)
         │
  base   ●──────── cubes (~x=0.35-0.40) ──── stack target (x=0.55)
  (0,0,0)  OpenArm root on lab-table top (z = 0)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

# ─────────────────────────────────────────────────────────────────────────────
# Scene geometry constants (official OpenArm lift layout)
# ─────────────────────────────────────────────────────────────────────────────

# The OpenArm lift scene places the ground plane well below the lab table so the
# table top surface lands at z = 0 in the local env frame (where the robot root
# and the manipulated object live).
GROUND_Z: float = -1.05
TABLE_TOP_Z: float = 0.0

# DexCube asset.  The upstream lift task spawns the object with scale 0.8.
CUBE_SCALE: float = 0.8
# Nominal edge length of the NVIDIA dex_cube_instanceable mesh (metres).
DEX_CUBE_BASE_SIZE: float = 0.08
# Effective cube edge length after scaling — used as the per-level stack height.
CUBE_HEIGHT: float = round(DEX_CUBE_BASE_SIZE * CUBE_SCALE, 6)  # 0.064 m
# Backwards-compatible alias used throughout the MDP / planner code.
CUBE_SIZE: float = CUBE_HEIGHT

# Spawn height the upstream OpenArm lift task uses for the object centre.
CUBE_SPAWN_Z: float = 0.055
# Backwards-compatible alias (used by eval scenarios / older imports).
CUBE_TABLE_Z: float = CUBE_SPAWN_Z

NUM_CUBES: int = 5
CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{i}" for i in range(NUM_CUBES))

# Cube colours (RGB) so the five stackable cubes are visually distinguishable.
CUBE_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.9, 0.1, 0.1),  # red
    (0.1, 0.4, 0.9),  # blue
    (0.1, 0.8, 0.2),  # green
    (0.9, 0.7, 0.1),  # yellow
    (0.6, 0.2, 0.8),  # purple
)

# Five cube spawn positions in the reachable area near the official lift cube
# spawn (≈ [0.4, 0.0, 0.055]).  Positions are in the LOCAL env frame.
CUBE_SPAWN_LOCAL_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.35, -0.16, CUBE_SPAWN_Z),
    (0.35, -0.08, CUBE_SPAWN_Z),
    (0.40, 0.00, CUBE_SPAWN_Z),
    (0.35, 0.08, CUBE_SPAWN_Z),
    (0.35, 0.16, CUBE_SPAWN_Z),
)

# Stack target: cubes are stacked above this XY in the same reachable workspace.
STACK_BASE_XY: tuple[float, float] = (0.55, 0.0)
STACK_BASE_Z: float = CUBE_SPAWN_Z
STACK_BASE_LOCAL_POS: tuple[float, float, float] = (*STACK_BASE_XY, STACK_BASE_Z)

# ── Backwards-compatible aliases (older imports expect these names) ──────────
DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS = CUBE_SPAWN_LOCAL_POSITIONS
DEFAULT_STACK_BASE_LOCAL_POS = STACK_BASE_LOCAL_POS
TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS = CUBE_SPAWN_LOCAL_POSITIONS
TABLETOP_STACK_BASE_LOCAL_POS = STACK_BASE_LOCAL_POS


# ─────────────────────────────────────────────────────────────────────────────
# EE debug markers
# ─────────────────────────────────────────────────────────────────────────────

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_EE_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: single stackable DexCube config
# ─────────────────────────────────────────────────────────────────────────────


def make_cube_cfg(
    name: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float] | None = None,
) -> RigidObjectCfg:
    """Return a :class:`RigidObjectCfg` for one stackable DexCube.

    The cube uses the same ``DexCube`` asset and scale (0.8) as the official
    OpenArm lift object.  An optional ``color`` recolours the cube so the five
    stack cubes remain visually distinct.
    """
    spawn = UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(CUBE_SCALE, CUBE_SCALE, CUBE_SCALE),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )
    if color is not None:
        spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0)

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=spawn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpenArm lift-style robot config (standard lift base pose)
# ─────────────────────────────────────────────────────────────────────────────

# Identical to the official OpenArm lift: the robot keeps its default base pose
# (root at the env origin, sitting on the lab-table top) — no custom raise.
OPENARM_LIFT_CFG: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ─────────────────────────────────────────────────────────────────────────────
# Reusable lift-style scene (robot + lab table + ground + light + EE frame)
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class OpenArmLiftStyleSceneCfg(InteractiveSceneCfg):
    """Base scene reproducing ``Isaac-Lift-Cube-OpenArm`` (no cubes).

    Sub-classes add the stackable cubes.
    """

    # OpenArm fixed at the standard lift base pose.
    robot: ArticulationCfg = OPENARM_LIFT_CFG

    # Standard Isaac Lab manipulation lab table (top surface at z = 0).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Ground plane dropped below the table so the table top sits at z = 0.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, GROUND_Z]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # EE frame transformer: source = robot base, target = TCP.
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


@configclass
class OpenArmLiftStyleWithCubesSceneCfg(OpenArmLiftStyleSceneCfg):
    """Lift-style scene plus five stackable DexCubes."""

    cube_0 = make_cube_cfg("Cube_0", CUBE_SPAWN_LOCAL_POSITIONS[0], CUBE_COLORS[0])
    cube_1 = make_cube_cfg("Cube_1", CUBE_SPAWN_LOCAL_POSITIONS[1], CUBE_COLORS[1])
    cube_2 = make_cube_cfg("Cube_2", CUBE_SPAWN_LOCAL_POSITIONS[2], CUBE_COLORS[2])
    cube_3 = make_cube_cfg("Cube_3", CUBE_SPAWN_LOCAL_POSITIONS[3], CUBE_COLORS[3])
    cube_4 = make_cube_cfg("Cube_4", CUBE_SPAWN_LOCAL_POSITIONS[4], CUBE_COLORS[4])