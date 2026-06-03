from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

##
# Shared tabletop geometry
##

NUM_CUBES = 5
CUBE_SIZE = 0.05
CUBE_NAMES = tuple(f"cube_{i}" for i in range(NUM_CUBES))

# TABLE_CENTER is the bottom-center of the simple workbench cuboid. Keeping the
# bottom at z=0 makes TABLE_TOP_Z equal to the cuboid height and keeps the ground
# plane below the table.
TABLE_CENTER = (0.55, 0.0, 0.0)
TABLE_SIZE = (0.95, 0.75, 0.20)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2]

# OpenArm's configured root frame is treated as the mount frame. If the asset
# origin is later found to be above or below the physical mounting surface,
# tune this single offset instead of changing scene spawn sites throughout the
# task configs.
OPENARM_BASE_Z_OFFSET = 0.0
ROBOT_BASE_ON_TABLE_Z = TABLE_TOP_Z + OPENARM_BASE_Z_OFFSET
ROBOT_BASE_ON_TABLE_POS = (TABLE_CENTER[0], TABLE_CENTER[1], ROBOT_BASE_ON_TABLE_Z)

# Cube centers rest half a cube above the workbench top. Stack target i is
# CUBE_TABLE_Z + i * CUBE_SIZE.
CUBE_TABLE_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0
DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS = (
    (0.34, -0.18, CUBE_TABLE_Z),
    (0.34, -0.09, CUBE_TABLE_Z),
    (0.34, 0.00, CUBE_TABLE_Z),
    (0.34, 0.09, CUBE_TABLE_Z),
    (0.34, 0.18, CUBE_TABLE_Z),
)
DEFAULT_STACK_BASE_LOCAL_POS = (0.55, 0.0, CUBE_TABLE_Z)

OPENARM_BASE_BODY = "openarm_link0"
OPENARM_EE_BODY = "openarm_ee_tcp"
OPENARM_ARM_JOINTS = ["openarm_joint.*"]
OPENARM_FINGER_JOINTS = ["openarm_finger_joint.*"]
OPENARM_GRIPPER_OPEN = 0.044
OPENARM_GRIPPER_CLOSED = 0.0

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)
_EE_MARKER_CFG.prim_path = "/Visuals/OpenArm/EndEffectorFrame"


def openarm_robot_cfg() -> ArticulationCfg:
    """Return OpenArm configured with its root mounted on the tabletop."""
    robot = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = ROBOT_BASE_ON_TABLE_POS
    return robot


def cube_target_z(cube_index: int) -> float:
    """Expected cube-center height for the sequential stack target."""
    return CUBE_TABLE_Z + cube_index * CUBE_SIZE


def cube_cfg(name: str, pos: tuple[float, float, float], color: tuple[float, float, float]) -> RigidObjectCfg:
    """Create one tabletop cube with consistent physics and visual properties."""
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
    """Shared scene: ground, workbench, OpenArm on the tabletop, five cubes, light, and EE marker."""

    robot: ArticulationCfg = openarm_robot_cfg()

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35), metallic=0.0),
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

    # The FrameTransformer exposes and visualizes the OpenArm TCP frame. Body and
    # prim names match OPENARM_UNI_CFG and the original baseline task.
    ee_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{OPENARM_BASE_BODY}",
        debug_vis=False,
        visualizer_cfg=_EE_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{OPENARM_EE_BODY}",
                name="end_effector",
            ),
        ],
    )

    cube_0 = cube_cfg("Cube_0", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[0], (0.9, 0.1, 0.1))
    cube_1 = cube_cfg("Cube_1", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[1], (0.1, 0.4, 0.9))
    cube_2 = cube_cfg("Cube_2", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[2], (0.1, 0.8, 0.2))
    cube_3 = cube_cfg("Cube_3", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[3], (0.9, 0.7, 0.1))
    cube_4 = cube_cfg("Cube_4", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[4], (0.6, 0.2, 0.8))
