from __future__ import annotations

import math

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..end_to_end import mdp as baseline_mdp
from ..tabletop_scene import (
    CUBE_NAMES,
    DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    DEFAULT_STACK_BASE_LOCAL_POS,
    OPENARM_ARM_JOINTS,
    OPENARM_EE_BODY,
    OPENARM_FINGER_JOINTS,
    OPENARM_GRIPPER_CLOSED,
    OPENARM_GRIPPER_OPEN,
    OpenArmTabletopSceneCfg,
)
from . import mdp


@configclass
class LLSceneCfg(OpenArmTabletopSceneCfg):
    """OpenArm on the workbench with five tabletop cubes for visual/task consistency."""


@configclass
class CommandsCfg:
    """Goal-conditioned EE pose target and per-episode gripper target."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=OPENARM_EE_BODY,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Ranges are in the OpenArm base frame. The base is mounted at
            # ROBOT_BASE_ON_TABLE_Z, so z>0 commands remain above the tabletop.
            pos_x=(-0.30, 0.20),
            pos_y=(-0.35, 0.35),
            pos_z=(0.05, 0.45),
            roll=(-0.35, 0.35),
            pitch=(2.6, math.pi),
            yaw=(-math.pi, math.pi),
        ),
    )

    grip_cmd = mdp.GripperCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        close_prob=0.5,
    )


@configclass
class ActionsCfg:
    """6D relative EE pose command plus binary OpenArm gripper command."""

    arm_action: ActionTerm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=OPENARM_ARM_JOINTS,
        body_name=OPENARM_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
    )

    gripper_action: ActionTerm = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_FINGER_JOINTS,
        open_command_expr={"openarm_finger_joint.*": OPENARM_GRIPPER_OPEN},
        close_command_expr={"openarm_finger_joint.*": OPENARM_GRIPPER_CLOSED},
    )


@configclass
class FallbackJointPositionActionsCfg:
    """Joint-position fallback preserving the LL goal-conditioned observation interface.

    Use this if an Isaac Lab/OpenArm asset variant does not expose
    ``openarm_ee_tcp`` as a body compatible with Differential IK.
    """

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_ARM_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=OPENARM_FINGER_JOINTS,
        open_command_expr={"openarm_finger_joint.*": OPENARM_GRIPPER_OPEN},
        close_command_expr={"openarm_finger_joint.*": OPENARM_GRIPPER_CLOSED},
    )


@configclass
class ObservationsCfg:
    """Concatenated LL policy observation."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose_b = ObsTerm(
            func=mdp.ee_pose_in_robot_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=OPENARM_EE_BODY)},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        grip_command = ObsTerm(func=mdp.grip_command_obs, params={"command_name": "grip_cmd"})
        gripper_pos = ObsTerm(
            func=mdp.gripper_pos_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_FINGER_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset robot, task target, and tabletop cubes for each LL episode."""

    reset_robot = EventTerm(
        func=baseline_mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_stack_target = EventTerm(
        func=baseline_mdp.reset_stack_target,
        mode="reset",
        params={"local_stack_base": DEFAULT_STACK_BASE_LOCAL_POS, "position_noise": 0.0},
    )

    reset_cubes = EventTerm(
        func=baseline_mdp.reset_cubes_non_overlapping,
        mode="reset",
        params={
            "cube_names": CUBE_NAMES,
            "local_positions": DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
            "position_noise": 0.0,
        },
    )


@configclass
class RewardsCfg:
    """Dense EE pose and gripper tracking rewards."""

    ee_pos_tracking_coarse = RewTerm(
        func=mdp.position_command_error,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=OPENARM_EE_BODY), "command_name": "ee_pose"},
    )
    ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={"std": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names=OPENARM_EE_BODY), "command_name": "ee_pose"},
    )
    ee_ori_tracking_coarse = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=OPENARM_EE_BODY), "command_name": "ee_pose"},
    )
    ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=0.5,
        params={"std": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names=OPENARM_EE_BODY), "command_name": "ee_pose"},
    )
    grip_tracking = RewTerm(
        func=mdp.gripper_command_tracking,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_FINGER_JOINTS), "command_name": "grip_cmd"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """LL terminates only on episode timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Ramp smoothness penalties during LL training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.03, "num_steps": 10_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.005, "num_steps": 10_000},
    )


@configclass
class OpenArmCubeStackLLEnvCfg(ManagerBasedRLEnvCfg):
    """Training config for the OpenArm LL goal-conditioned EE tracker."""

    scene: LLSceneCfg = LLSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 2.5)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
