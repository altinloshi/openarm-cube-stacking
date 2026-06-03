"""Shared constants and reusable scene configuration for OpenArm tabletop tasks.

Layout (local env frame, with env origins added at runtime):

         +x (forward)
         │
  env    ●────────────────────────────── table (0.55 m ahead, 0.85×0.70 m)
  origin (0,0)                robot mounted ON TABLE TOP at z=TABLE_TOP_Z

Robot base is at (ROBOT_ON_TABLE_X, ROBOT_ON_TABLE_Y, ROBOT_ON_TABLE_Z).
Cubes are placed in front of the robot on the table surface.
"""

from __future__ import annotations


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

# ─────────────────────────────────────────────────────────────────────────────
# Table geometry constants
# ─────────────────────────────────────────────────────────────────────────────

# Table cuboid height.  The table spawn position is at (cx, cy, TABLE_HEIGHT/2)
# so that the TOP SURFACE sits exactly at TABLE_TOP_Z.
TABLE_HEIGHT: float = 0.20  # metres

# Z of the table's top surface in the local environment frame
TABLE_TOP_Z: float = TABLE_HEIGHT  # = 0.20 m

TABLE_CENTER_X: float = 0.55   # table centre x in local env frame
TABLE_CENTER_Y: float = 0.0
TABLE_SIZE_X: float = 0.85     # table length along x
TABLE_SIZE_Y: float = 0.70     # table width along y

# ─────────────────────────────────────────────────────────────────────────────
# Robot placement on the table
# ─────────────────────────────────────────────────────────────────────────────

# If the USD root origin of the OpenArm model is NOT at its physical base plate
# bottom, adjust this offset so the robot sits flush on the table.
# Set to 0.0 if the root is already at the bottom of the robot body.
OPENARM_BASE_Z_OFFSET: float = 0.0

# Final position of the robot base in the local environment frame
ROBOT_ON_TABLE_X: float = TABLE_CENTER_X  # 0.55 m
ROBOT_ON_TABLE_Y: float = TABLE_CENTER_Y  # 0.00 m
ROBOT_ON_TABLE_Z: float = TABLE_TOP_Z + OPENARM_BASE_Z_OFFSET  # 0.20 m

# ─────────────────────────────────────────────────────────────────────────────
# Cube parameters
# ─────────────────────────────────────────────────────────────────────────────

CUBE_SIZE: float = 0.05   # cube edge length in metres
NUM_CUBES: int = 5

# Z of a cube's centre when it rests on the table
CUBE_TABLE_Z: float = TABLE_TOP_Z + CUBE_SIZE / 2.0  # = 0.225 m

CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{i}" for i in range(NUM_CUBES))

# Cube colours (RGB)
CUBE_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.9, 0.1, 0.1),  # red
    (0.1, 0.4, 0.9),  # blue
    (0.1, 0.8, 0.2),  # green
    (0.9, 0.7, 0.1),  # yellow
    (0.6, 0.2, 0.8),  # purple
)

# Default cube spawn positions for the TABLETOP scene.
# Cubes are placed in front of the robot (x > ROBOT_ON_TABLE_X).
TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.75, -0.18, CUBE_TABLE_Z),
    (0.75, -0.09, CUBE_TABLE_Z),
    (0.75,  0.00, CUBE_TABLE_Z),
    (0.75,  0.09, CUBE_TABLE_Z),
    (0.75,  0.18, CUBE_TABLE_Z),
)

# Stack target position: cubes are stacked above this XY on the table
TABLETOP_STACK_BASE_LOCAL_POS: tuple[float, float, float] = (0.85, 0.0, CUBE_TABLE_Z)

# ─────────────────────────────────────────────────────────────────────────────
# EE debug marker
# ─────────────────────────────────────────────────────────────────────────────

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)
_EE_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"

_EE_MARKER_CFG_VIS = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG_VIS.markers["frame"].scale = (0.10, 0.10, 0.10)
_EE_MARKER_CFG_VIS.prim_path = "/Visuals/EETarget"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: single cube config
# ─────────────────────────────────────────────────────────────────────────────

def make_cube_cfg(
    name: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Return a RigidObjectCfg for a single stackable cube."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Reusable tabletop robot config (robot mounted on table)
# ─────────────────────────────────────────────────────────────────────────────

# Robot articulation config with the base placed on the table top.
# init_state.pos is in the LOCAL environment frame; env_origins are added at
# reset time by the reset_robot_to_default event.
OPENARM_ON_TABLE_CFG: ArticulationCfg = OPENARM_UNI_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        # Robot base sits on the table surface at the table centre
        pos=(ROBOT_ON_TABLE_X, ROBOT_ON_TABLE_Y, ROBOT_ON_TABLE_Z),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Arm joints – same as OPENARM_UNI_CFG defaults
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            # Finger joints fully open
            "openarm_finger_joint.*": 0.044,
        },
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable tabletop scene config (robot + table + ground + light + EE frame)
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class OpenArmTabletopSceneCfg(InteractiveSceneCfg):
    """Base scene: OpenArm mounted on a table, no cubes.

    Sub-classes (LL, HL, Eval) add cubes and/or additional assets.
    """

    # Robot mounted on the table top
    robot: ArticulationCfg = OPENARM_ON_TABLE_CFG

    # Table: cuboid centered at (TABLE_CENTER_X, 0, TABLE_HEIGHT/2)
    # so that the top surface is at z = TABLE_HEIGHT = TABLE_TOP_Z
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_HEIGHT / 2.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.35), metallic=0.0
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # EE frame transformer: source = robot base, target = TCP
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
class OpenArmTabletopWithCubesSceneCfg(OpenArmTabletopSceneCfg):
    """Full tabletop scene: robot + table + five stackable cubes."""

    cube_0 = make_cube_cfg("Cube_0", TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[0], CUBE_COLORS[0])
    cube_1 = make_cube_cfg("Cube_1", TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[1], CUBE_COLORS[1])
    cube_2 = make_cube_cfg("Cube_2", TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[2], CUBE_COLORS[2])
    cube_3 = make_cube_cfg("Cube_3", TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[3], CUBE_COLORS[3])
    cube_4 = make_cube_cfg("Cube_4", TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[4], CUBE_COLORS[4])
