"""Manager-based environment configuration for OpenArm five-cube stacking."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp


CUBE_SIZE = 0.045
TABLE_TOP_Z = 0.40


def _cube_cfg(name: str, color: tuple[float, float, float], position: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(position), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.1),
        ),
    )


@configclass
class OpenArmCubeStackSceneCfg(InteractiveSceneCfg):
    """Interactive scene with an OpenArm robot, table, and five cubes."""

    robot = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, 0.0, 0.2]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.33, 0.30), roughness=0.6),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
    )

    cube_0 = _cube_cfg("Cube_0", (0.85, 0.10, 0.10), (0.42, -0.16, TABLE_TOP_Z + 0.5 * CUBE_SIZE))
    cube_1 = _cube_cfg("Cube_1", (0.10, 0.55, 0.85), (0.48, -0.05, TABLE_TOP_Z + 0.5 * CUBE_SIZE))
    cube_2 = _cube_cfg("Cube_2", (0.15, 0.75, 0.25), (0.56, 0.14, TABLE_TOP_Z + 0.5 * CUBE_SIZE))
    cube_3 = _cube_cfg("Cube_3", (0.85, 0.65, 0.10), (0.64, 0.02, TABLE_TOP_Z + 0.5 * CUBE_SIZE))
    cube_4 = _cube_cfg("Cube_4", (0.75, 0.20, 0.75), (0.68, -0.12, TABLE_TOP_Z + 0.5 * CUBE_SIZE))

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )


@configclass
class ActionsCfg:
    """Action specification for OpenArm joint and gripper control."""

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
    """Observation groups for policy learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        ee_pose = ObsTerm(func=mdp.ee_pose)
        current_cube_position = ObsTerm(func=mdp.current_cube_position)
        cube_positions = ObsTerm(func=mdp.cube_positions)
        cube_orientations = ObsTerm(func=mdp.cube_orientations)
        current_cube_index = ObsTerm(func=mdp.current_cube_index)
        current_target_position = ObsTerm(func=mdp.current_target_position)
        ee_to_current_cube = ObsTerm(func=mdp.ee_to_current_cube)
        current_cube_to_target = ObsTerm(func=mdp.current_cube_to_target)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for robot, cubes, and the stack target."""

    reset_robot = EventTerm(func=mdp.reset_robot_to_default, mode="reset")
    reset_stack_target = EventTerm(func=mdp.reset_stack_target, mode="reset")
    reset_cubes = EventTerm(func=mdp.reset_cubes_non_overlapping, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for sequential cube stacking."""

    reaching_current_cube = RewTerm(func=mdp.reaching_current_cube, weight=1.5)
    lifting_current_cube = RewTerm(func=mdp.lifting_current_cube, weight=4.0)
    moving_current_cube_to_target = RewTerm(func=mdp.moving_current_cube_to_target, weight=6.0)
    placing_current_cube = RewTerm(func=mdp.placing_current_cube, weight=10.0)
    upright_stack_bonus = RewTerm(func=mdp.upright_stack_bonus, weight=2.0)
    stack_success_bonus = RewTerm(func=mdp.stack_success_bonus, weight=30.0)
    action_penalty = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cube_drop_penalty = RewTerm(func=mdp.cube_drop_penalty, weight=-8.0)
    stack_collapse_penalty = RewTerm(func=mdp.stack_collapse_penalty, weight=-6.0)


@configclass
class TerminationsCfg:
    """Termination terms for the stacking task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    all_cubes_stacked = DoneTerm(func=mdp.all_cubes_stacked)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    stack_collapsed = DoneTerm(func=mdp.stack_collapsed)
    invalid_state = DoneTerm(func=mdp.invalid_state)


@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment for stacking five cubes with OpenArm."""

    scene: OpenArmCubeStackSceneCfg = OpenArmCubeStackSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 12.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
