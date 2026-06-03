from __future__ import annotations

"""Reusable tabletop scene configuration for the OpenArm cube-stacking pipeline.

This module centralises every geometric constant (table height, robot base
offset, cube size, cube spawn layout, stack-target location) so the whole
hierarchical pipeline (``ll_policy``, ``hl_policy`` and ``eval``) shares one
consistent tabletop layout. The ``end_to_end`` baseline keeps its own scene.

Visual layout (matches the tournament reference screenshot)
-----------------------------------------------------------
* A ground plane sits below the workbench.
* One large table/workbench is spawned per environment.
* The OpenArm robot is mounted **on top of the table** (its base origin sits at
  the tabletop surface, not on the world floor).
* Five cubes are spawned on the same tabletop surface in front of the robot.
* A dome light illuminates the scene.
* A frame-transformer drives the end-effector debug marker.

Body / joint names (must match ``OPENARM_UNI_CFG`` and the current repo)
-----------------------------------------------------------------------
* Arm joints       : ``openarm_joint[1-7]`` (matched by ``openarm_joint.*``)
* Gripper joints   : ``openarm_finger_joint.*`` (open = 0.044 m, close = 0.0 m)
* Base link        : ``openarm_link0``
* End-effector body: ``openarm_hand``
* TCP debug frame  : ``openarm_ee_tcp``
"""

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

##
# Geometry constants (single source of truth for the tabletop pipeline)
##

# Cube edge length (m). Matches the cube size used by the existing end-to-end task.
CUBE_SIZE: float = 0.05
NUM_CUBES: int = 5

# Table/workbench dimensions. The cuboid table is centred at ``TABLE_CENTER`` with
# its bottom on the floor, so the tabletop surface is at ``TABLE_TOP_Z``.
TABLE_CENTER: tuple[float, float, float] = (0.55, 0.0, 0.0)
TABLE_LENGTH_X: float = 1.20
TABLE_WIDTH_Y: float = 0.90
TABLE_HEIGHT: float = 0.20
# The table cuboid is placed with its centre at height ``TABLE_HEIGHT / 2`` so that
# the top surface (where everything rests) is exactly at ``TABLE_TOP_Z``.
TABLE_TOP_Z: float = TABLE_HEIGHT

# OpenArm base offset above the tabletop. ``openarm_link0`` (the base link origin)
# is assumed to coincide with the physical bottom of the robot base, so no extra
# lift is required. If the robot visibly penetrates or floats above the table,
# raise/lower this single constant rather than editing scattered numbers.
OPENARM_BASE_Z_OFFSET: float = 0.0
ROBOT_BASE_ON_TABLE_Z: float = TABLE_TOP_Z + OPENARM_BASE_Z_OFFSET
# Robot base is mounted on top of the table, centred at the table centre XY.
ROBOT_BASE_ON_TABLE_POS: tuple[float, float, float] = (
    TABLE_CENTER[0],
    TABLE_CENTER[1],
    ROBOT_BASE_ON_TABLE_Z,
)

# A cube resting on the tabletop has its centre half a cube above the surface.
CUBE_TABLE_Z: float = TABLE_TOP_Z + CUBE_SIZE / 2.0

# Default cube spawn layout, expressed in the per-environment local frame
# (``env_origins`` are added at runtime). Cubes are laid out in a row in front of
# the robot base (smaller X = toward the viewer) and within arm reach.
DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.32, -0.18, CUBE_TABLE_Z),
    (0.32, -0.09, CUBE_TABLE_Z),
    (0.32, 0.00, CUBE_TABLE_Z),
    (0.32, 0.09, CUBE_TABLE_Z),
    (0.32, 0.18, CUBE_TABLE_Z),
)

# Default stack-base XY on the tabletop (local frame). Kept clear of the robot
# base and the cube spawn row, and within arm reach.
DEFAULT_STACK_BASE_LOCAL_POS: tuple[float, float, float] = (0.44, 0.0, CUBE_TABLE_Z)

CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{i}" for i in range(NUM_CUBES))

# Per-cube colours (for clarity in the viewport).
CUBE_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.9, 0.1, 0.1),
    (0.1, 0.4, 0.9),
    (0.1, 0.8, 0.2),
    (0.9, 0.7, 0.1),
    (0.6, 0.2, 0.8),
)


def _ee_marker_cfg():
    """Frame marker config for the end-effector debug visualisation."""
    cfg = FRAME_MARKER_CFG.copy()
    cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    cfg.prim_path = "/Visuals/EEFrameTransformer"
    return cfg


def make_tabletop_robot(high_pd: bool = False) -> ArticulationCfg:
    """Return an OpenArm articulation config mounted on top of the table.

    The robot base origin (``openarm_link0``) is placed at
    ``ROBOT_BASE_ON_TABLE_POS`` so the arm sits on the tabletop. A deep copy is
    used so the shared ``OPENARM_UNI_CFG`` (and its ``init_state``) is never
    mutated.

    Args:
        high_pd: If ``True`` use ``OPENARM_UNI_HIGH_PD_CFG`` (stiffer gains, gravity
            disabled) which is required for accurate differential-IK tracking.
            Imported lazily so callers that do not need IK avoid the dependency.
    """
    if high_pd:
        from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG

        base = OPENARM_UNI_HIGH_PD_CFG
    else:
        base = OPENARM_UNI_CFG

    robot = copy.deepcopy(base)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    # Mount the base on the tabletop. The joint_pos defaults are preserved.
    robot.init_state.pos = ROBOT_BASE_ON_TABLE_POS
    return robot


def make_cube_cfg(
    name: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Create one stackable cube with consistent physics properties."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0),
        ),
    )


@configclass
class OpenArmTabletopSceneCfg(InteractiveSceneCfg):
    """Reusable tabletop scene: OpenArm mounted on a workbench.

    Contains the ground plane, the table/workbench, the OpenArm robot mounted on
    top of the table, a dome light, and the end-effector debug frame transformer.
    Cube objects are added by :class:`OpenArmTabletopStackSceneCfg` for the
    stacking environments (HL / eval); the bare scene is used by the low-level
    EE-tracking policy which does not need cubes.
    """

    # OpenArm robot mounted on the tabletop (joint-position control variant by
    # default; envs that need differential IK swap in the high-PD variant in
    # their ``__post_init__``).
    robot: ArticulationCfg = make_tabletop_robot(high_pd=False)

    # Large table/workbench, one per environment. Bottom on the floor, top at
    # ``TABLE_TOP_Z``.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_HEIGHT / 2.0)),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_LENGTH_X, TABLE_WIDTH_Y, TABLE_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), metallic=0.0),
        ),
    )

    # Ground plane sits just below the table bottom.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # End-effector debug frame: tracks the TCP frame relative to the base link.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=_ee_marker_cfg(),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )


@configclass
class OpenArmTabletopStackSceneCfg(OpenArmTabletopSceneCfg):
    """Tabletop scene extended with five stackable cubes on the tabletop."""

    cube_0 = make_cube_cfg("Cube_0", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[0], CUBE_COLORS[0])
    cube_1 = make_cube_cfg("Cube_1", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[1], CUBE_COLORS[1])
    cube_2 = make_cube_cfg("Cube_2", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[2], CUBE_COLORS[2])
    cube_3 = make_cube_cfg("Cube_3", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[3], CUBE_COLORS[3])
    cube_4 = make_cube_cfg("Cube_4", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[4], CUBE_COLORS[4])
