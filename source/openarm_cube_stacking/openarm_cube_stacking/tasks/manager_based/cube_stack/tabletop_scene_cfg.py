"""Shared OpenArm CubeStack scene using the official OpenArm lift-table setup.

This scene intentionally matches Isaac-Lift-Cube-OpenArm-Play-v0:
- SeattleLabTable workbench
- OpenArm mounted using OPENARM_UNI_CFG
- DexCube objects in the official OpenArm reachable workspace
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from isaaclab.markers.config import FRAME_MARKER_CFG


# ---------------------------------------------------------------------
# Constants used by CubeStack MDP code
# ---------------------------------------------------------------------

CUBE_NAMES = ["cube_0", "cube_1", "cube_2", "cube_3", "cube_4"]
NUM_CUBES: int = len(CUBE_NAMES)

# DexCube is spawned at z=0.055 in the official OpenArm lift setup.
# Keep this value so the cubes sit in the same workspace as Isaac-Lift-Cube-OpenArm.
CUBE_TABLE_Z: float = 0.055

# Approximate scaled DexCube size. This is used by rewards/planner target heights.
CUBE_SIZE: float = 0.05

# Table top used by reward/planner logic. Since cube center is 0.055,
# table contact height is approximately cube_center - cube_size / 2.
TABLE_TOP_Z: float = CUBE_TABLE_Z - CUBE_SIZE / 2.0

# Kept only for old imports. Do not use this for robot placement.
ROBOT_ON_TABLE_X: float = 0.0
ROBOT_ON_TABLE_Y: float = 0.0
ROBOT_ON_TABLE_Z: float = 0.0

TABLETOP_STACK_BASE_LOCAL_POS = (0.55, 0.0, CUBE_TABLE_Z)

# ---------------------------------------------------------------------
# Backward-compatible names for older LL/HL configs.
# These do NOT create a custom cuboid table. They only prevent old imports
# from crashing while the scene uses the official SeattleLabTable asset.
# ---------------------------------------------------------------------

TABLE_CENTER_X: float = 0.5
TABLE_CENTER_Y: float = 0.0
TABLE_SIZE_X: float = 0.85
TABLE_SIZE_Y: float = 0.70

# Kept only for compatibility with older imports.
# Do not use this to spawn a cuboid table.
TABLE_HEIGHT: float = 0.0

ROBOT_BASE_ON_TABLE_X: float = ROBOT_ON_TABLE_X
ROBOT_BASE_ON_TABLE_Y: float = ROBOT_ON_TABLE_Y
ROBOT_BASE_ON_TABLE_Z: float = ROBOT_ON_TABLE_Z
ROBOT_BASE_ON_TABLE_POS = (ROBOT_BASE_ON_TABLE_X, ROBOT_BASE_ON_TABLE_Y, ROBOT_BASE_ON_TABLE_Z)


CUBE_INITIAL_LOCAL_POSITIONS = {
    "cube_0": (0.35, -0.16, CUBE_TABLE_Z),
    "cube_1": (0.35, -0.08, CUBE_TABLE_Z),
    "cube_2": (0.40,  0.00, CUBE_TABLE_Z),
    "cube_3": (0.35,  0.08, CUBE_TABLE_Z),
    "cube_4": (0.35,  0.16, CUBE_TABLE_Z),
}

# Backward-compatible aliases for older HL/LL modules.
# These names point to the new official-lift-style cube positions.
TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS = CUBE_INITIAL_LOCAL_POSITIONS
TABLETOP_CUBE_SPAWN_POSITIONS = CUBE_INITIAL_LOCAL_POSITIONS
TABLETOP_CUBE_INITIAL_LOCAL_POSITIONS = CUBE_INITIAL_LOCAL_POSITIONS
STACK_BASE_LOCAL_POS = TABLETOP_STACK_BASE_LOCAL_POS
STACK_TARGET_LOCAL_POS = TABLETOP_STACK_BASE_LOCAL_POS



def _make_dex_cube(name: str, pos: tuple[float, float, float]) -> RigidObjectCfg:
    """Create one DexCube object matching the official OpenArm lift object asset."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
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


# Official OpenArm EE frame marker setup.
_marker_cfg = FRAME_MARKER_CFG.copy()
_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
_marker_cfg.prim_path = "/Visuals/FrameTransformer"


@configclass
class OpenArmTabletopSceneCfg(InteractiveSceneCfg):
    """Official OpenArm lift-style scene with SeattleLabTable and OpenArm."""

    # OpenArm robot: do not manually raise it onto a fake platform.
    robot: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Official Isaac-Lift-Cube-OpenArm table.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Official ground plane.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Official light.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Official OpenArm EE frame names.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=True,
        visualizer_cfg=_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )


@configclass
class OpenArmTabletopWithCubesSceneCfg(OpenArmTabletopSceneCfg):
    """OpenArm lift-style scene with five DexCube objects for CubeStack."""

    cube_0: RigidObjectCfg = _make_dex_cube("cube_0", CUBE_INITIAL_LOCAL_POSITIONS["cube_0"])
    cube_1: RigidObjectCfg = _make_dex_cube("cube_1", CUBE_INITIAL_LOCAL_POSITIONS["cube_1"])
    cube_2: RigidObjectCfg = _make_dex_cube("cube_2", CUBE_INITIAL_LOCAL_POSITIONS["cube_2"])
    cube_3: RigidObjectCfg = _make_dex_cube("cube_3", CUBE_INITIAL_LOCAL_POSITIONS["cube_3"])
    cube_4: RigidObjectCfg = _make_dex_cube("cube_4", CUBE_INITIAL_LOCAL_POSITIONS["cube_4"])
