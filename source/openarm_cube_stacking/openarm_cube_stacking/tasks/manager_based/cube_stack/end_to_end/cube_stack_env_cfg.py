from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..tabletop_scene import (
    CUBE_NAMES,
    DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    DEFAULT_STACK_BASE_LOCAL_POS,
    OPENARM_ARM_JOINTS,
    OPENARM_FINGER_JOINTS,
    OPENARM_GRIPPER_CLOSED,
    OPENARM_GRIPPER_OPEN,
    OpenArmTabletopSceneCfg,
)
from . import mdp


@configclass
class OpenArmCubeStackSceneCfg(OpenArmTabletopSceneCfg):
    """End-to-end scene: OpenArm is mounted on the workbench with five tabletop cubes."""


@configclass
class ActionsCfg:
    """Original OpenArm joint-position arm control plus binary finger control."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_FINGER_JOINTS,
        open_command_expr={"openarm_finger_joint.*": OPENARM_GRIPPER_OPEN},
        close_command_expr={"openarm_finger_joint.*": OPENARM_GRIPPER_CLOSED},
    )


@configclass
class ObservationsCfg:
    """Observation terms for the original end-to-end stacking policy."""

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
    """Reward terms for sequential five-cube end-to-end stacking."""

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
    """Termination terms for the end-to-end task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    all_five_cubes_stacked = DoneTerm(func=mdp.all_cubes_stacked, params={"threshold": 0.035})
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    stack_collapsed = DoneTerm(func=mdp.stack_collapsed)


@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based end-to-end baseline config."""

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
