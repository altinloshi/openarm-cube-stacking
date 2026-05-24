from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from . import mdp
from .mdp.observations import (
    CUBE_NAMES,
    CUBE_SIZE,
    DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    DEFAULT_STACK_BASE_LOCAL_POS,
    TABLE_HEIGHT,
)

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)
_EE_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


def _cube_cfg(name: str, pos: tuple[float, float, float], color: tuple[float, float, float]) -> RigidObjectCfg:
    """Create one cube config with consistent physics properties."""
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


##
# Scene definition
##


@configclass
class OpenArmCubeStackSceneCfg(InteractiveSceneCfg):
    """Scene with OpenArm, a table, and five stackable cubes."""

    robot: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, TABLE_HEIGHT / 2.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.85, 0.70, TABLE_HEIGHT),
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

    cube_0 = _cube_cfg("Cube_0", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[0], (0.9, 0.1, 0.1))
    cube_1 = _cube_cfg("Cube_1", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[1], (0.1, 0.4, 0.9))
    cube_2 = _cube_cfg("Cube_2", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[2], (0.1, 0.8, 0.2))
    cube_3 = _cube_cfg("Cube_3", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[3], (0.9, 0.7, 0.1))
    cube_4 = _cube_cfg("Cube_4", DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS[4], (0.6, 0.2, 0.8))


##
# MDP settings
##


@configclass
class ActionsCfg:
    """OpenArm joint-position arm control plus binary finger control."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation terms for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        ee_position = ObsTerm(func=mdp.end_effector_position)
        ee_orientation = ObsTerm(func=mdp.end_effector_orientation)
        cube_positions = ObsTerm(func=mdp.cube_positions)
        cube_orientations = ObsTerm(func=mdp.cube_orientations)
        current_cube_index = ObsTerm(func=mdp.current_cube_index)
        current_cube_position = ObsTerm(func=mdp.current_cube_position)
        current_target_position = ObsTerm(func=mdp.current_target_position)
        ee_to_current_cube = ObsTerm(func=mdp.ee_to_current_cube)
        current_cube_to_target = ObsTerm(func=mdp.current_cube_to_target)
        stack_target_positions = ObsTerm(func=mdp.stack_target_positions)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for robot, cubes, and task target."""

    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_stack_target = EventTerm(
        func=mdp.reset_stack_target,
        mode="reset",
        params={
            "local_stack_base": DEFAULT_STACK_BASE_LOCAL_POS,
            "position_noise": 0.02,
        },
    )

    reset_cubes = EventTerm(
        func=mdp.reset_cubes_non_overlapping,
        mode="reset",
        params={
            "cube_names": CUBE_NAMES,
            "local_positions": DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
            "position_noise": 0.015,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for sequential five-cube stacking."""

    reaching_current_cube = RewTerm(func=mdp.reaching_current_cube, weight=2.0, params={"std": 0.08})
    lifting_current_cube = RewTerm(func=mdp.lifting_current_cube, weight=4.0, params={"minimal_height": 0.04})
    moving_current_cube_to_target = RewTerm(
        func=mdp.moving_current_cube_to_target,
        weight=8.0,
        params={"std": 0.20, "minimal_height": 0.02},
    )
    placing_current_cube = RewTerm(func=mdp.placing_current_cube, weight=10.0, params={"threshold": 0.03})
    stack_success_bonus = RewTerm(func=mdp.stack_success_bonus, weight=25.0, params={"threshold": 0.035})
    cube_drop_penalty = RewTerm(func=mdp.cube_drop_penalty, weight=-5.0)
    stack_collapse_penalty = RewTerm(func=mdp.stack_collapse_penalty, weight=-5.0)
    action_penalty = RewTerm(func=mdp.action_l2, weight=-1.0e-4)
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_velocity_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    all_five_cubes_stacked = DoneTerm(func=mdp.all_cubes_stacked, params={"threshold": 0.035})
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    stack_collapsed = DoneTerm(func=mdp.stack_collapsed)


##
# Environment configuration
##


@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL config for ``Nepher-OpenArm-CubeStack-v0``."""

    scene: OpenArmCubeStackSceneCfg = OpenArmCubeStackSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

