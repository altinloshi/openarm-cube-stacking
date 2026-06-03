"""Shared tabletop scene configuration matching the official OpenArm lift setup.

The layout mirrors ``Isaac-Lift-Cube-OpenArm-Play-v0``: the OpenArm robot uses
the official asset root pose, the standard black SeattleLabTable USD is spawned
in front of it, and DexCube objects sit at the lift task's cube height.
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

# Official OpenArm lift-cube object settings.
DEX_CUBE_USD_PATH: str = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
DEX_CUBE_SCALE: tuple[float, float, float] = (0.8, 0.8, 0.8)

NUM_CUBES: int = 5
CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{i}" for i in range(NUM_CUBES))

# The official OpenArm lift task spawns the scaled DexCube with its center at z=0.055.
CUBE_SIZE: float = 0.05
CUBE_HEIGHT: float = CUBE_SIZE
CUBE_TABLE_Z: float = 0.055
CUBE_CENTER_Z: float = CUBE_TABLE_Z
TABLE_TOP_Z: float = CUBE_TABLE_Z - CUBE_SIZE / 2.0

TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.35, -0.16, CUBE_TABLE_Z),
    (0.35, -0.08, CUBE_TABLE_Z),
    (0.40, 0.00, CUBE_TABLE_Z),
    (0.35, 0.08, CUBE_TABLE_Z),
    (0.35, 0.16, CUBE_TABLE_Z),
)
TABLETOP_STACK_BASE_LOCAL_POS: tuple[float, float, float] = (0.55, 0.0, CUBE_TABLE_Z)

# Backwards-compatible names used by existing cube-stack modules.
OPENARM_LIFT_CUBE_SPAWN_LOCAL_POSITIONS = TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS
OPENARM_LIFT_STACK_BASE_LOCAL_POS = TABLETOP_STACK_BASE_LOCAL_POS

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_EE_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


def make_dex_cube_cfg(name: str, pos: tuple[float, float, float]) -> RigidObjectCfg:
    """Return a DexCube rigid object matching the official OpenArm lift task."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=DEX_CUBE_USD_PATH,
            scale=DEX_CUBE_SCALE,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )


@configclass
class OpenArmTabletopSceneCfg(InteractiveSceneCfg):
    """Official OpenArm lift scene layout with robot, SeattleLabTable, and EE frame."""

    robot: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

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
    """Official OpenArm lift-style tabletop scene with five DexCube objects."""

    cube_0 = make_dex_cube_cfg(CUBE_NAMES[0], TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[0])
    cube_1 = make_dex_cube_cfg(CUBE_NAMES[1], TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[1])
    cube_2 = make_dex_cube_cfg(CUBE_NAMES[2], TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[2])
    cube_3 = make_dex_cube_cfg(CUBE_NAMES[3], TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[3])
    cube_4 = make_dex_cube_cfg(CUBE_NAMES[4], TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS[4])


# Compatibility aliases for modules that adopted the lift-style scene names.
OpenArmLiftStyleSceneCfg = OpenArmTabletopSceneCfg
OpenArmLiftStyleWithCubesSceneCfg = OpenArmTabletopWithCubesSceneCfg
