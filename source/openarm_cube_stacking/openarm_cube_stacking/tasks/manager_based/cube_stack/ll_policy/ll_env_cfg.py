from __future__ import annotations

"""Low-level (LL) goal-conditioned EE-tracking environment for OpenArm.

The LL policy receives a continuously resampled target end-effector pose plus a
binary gripper command and learns to:
  1. Move the EE (``openarm_hand``) to the commanded pose (position + orientation).
  2. Open / close the gripper on command.

It deliberately does **not** learn cube stacking — that is the High-Level
classical planner's job. The LL policy is a reactive executor.

Action space (7D, differential-IK mode):
    arm_action   DifferentialIK delta pose (Δx, Δy, Δz, Δrx, Δry, Δrz)   [6]
    gripper      binary open/close command                               [1]

Observation space (concatenated):
    joint_pos    arm + finger joint positions (relative to default)
    joint_vel    arm + finger joint velocities
    ee_pose_b    current EE pose (pos + quat) in robot base frame        [7]
    pose_command target EE pose (pos + quat) from the command manager     [7]
    grip_command target gripper state (0 = open, 1 = close)               [1]
    gripper_pos  current normalised gripper opening [0, 1]                [1]
    actions      last applied action

Commands:
    ee_pose   UniformPoseCommandCfg — resampled every 4 s mid-episode
    grip_cmd  GripperCommandCfg     — resampled once per episode (50/50 open/close)
"""

import math

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
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

from .. import tabletop
from ..tabletop import OpenArmTabletopSceneCfg, make_tabletop_robot
from . import mdp

# ---------------------------------------------------------------------------
# Control-mode switch
# ---------------------------------------------------------------------------
# OpenArm exposes the body (``openarm_hand``) and arm joints (``openarm_joint.*``)
# required by differential IK, and ships a stiff high-PD variant
# (``OPENARM_UNI_HIGH_PD_CFG``) suited to task-space tracking, so differential IK
# is the default. Set this to ``False`` to fall back to direct joint-position
# control while keeping the LL interface goal-conditioned (the policy still
# observes the target EE pose and a separate binary gripper command).
USE_DIFFERENTIAL_IK: bool = True

# End-effector body used for tracking, commands and IK control.
EE_BODY_NAME: str = "openarm_hand"
ARM_JOINT_EXPR: str = "openarm_joint.*"
FINGER_JOINT_EXPR: str = "openarm_finger_joint.*"
GRIPPER_OPEN: float = 0.044
GRIPPER_CLOSE: float = 0.0

# IK control point relative to ``openarm_hand``. Zero keeps the controlled frame
# identical to the observed/commanded frame; the HL planner accounts for the
# hand-to-grasp offset separately (see classical_stack_planner.py).
IK_BODY_OFFSET = (0.0, 0.0, 0.0)


##
# Scene
##


@configclass
class LLSceneCfg(OpenArmTabletopSceneCfg):
    """Tabletop scene for LL EE-tracking (no cubes needed).

    Uses the high-PD OpenArm variant which is required for accurate IK tracking.
    """

    robot = make_tabletop_robot(high_pd=USE_DIFFERENTIAL_IK)


##
# Commands
##


@configclass
class CommandsCfg:
    """Random EE pose target (resampled mid-episode) + per-episode gripper target."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY_NAME,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Reachable workspace above the tabletop, expressed in the robot base
            # frame. These ranges are a sensible default for the OpenArm mounted
            # on the table; tune them to the measured reachable set if needed.
            pos_x=(0.10, 0.45),
            pos_y=(-0.30, 0.30),
            pos_z=(0.05, 0.40),
            # Orientation: gripper pointing mostly down with limited roll and full
            # yaw freedom (pitch ~ pi => tool z-axis points at the table).
            roll=(-0.3, 0.3),
            pitch=(2.8, math.pi),
            yaw=(-math.pi, math.pi),
        ),
    )

    # Resampled only at episode reset (timer far beyond any episode length).
    grip_cmd = mdp.GripperCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        close_prob=0.5,
    )


##
# Actions
##


@configclass
class ActionsCfg:
    """Differential-IK 6D arm delta + binary 1D gripper = 7D total action."""

    arm_action: ActionTerm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[ARM_JOINT_EXPR],
        body_name=EE_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=list(IK_BODY_OFFSET)),
    )

    gripper_action: ActionTerm = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[FINGER_JOINT_EXPR],
        open_command_expr={FINGER_JOINT_EXPR: GRIPPER_OPEN},
        close_command_expr={FINGER_JOINT_EXPR: GRIPPER_CLOSE},
    )


@configclass
class JointPositionActionsCfg:
    """Fallback: direct joint-position arm control + binary gripper.

    Kept available for environments where differential IK is undesirable. The
    LL interface stays goal-conditioned because the policy still observes the
    commanded EE pose and the binary gripper command.
    """

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[ARM_JOINT_EXPR],
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action: ActionTerm = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[FINGER_JOINT_EXPR],
        open_command_expr={FINGER_JOINT_EXPR: GRIPPER_OPEN},
        close_command_expr={FINGER_JOINT_EXPR: GRIPPER_CLOSE},
    )


##
# Observations
##


@configclass
class ObservationsCfg:
    """Fully-concatenated policy observation."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        ee_pose_b = ObsTerm(
            func=mdp.ee_pose_in_robot_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
        )

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        grip_command = ObsTerm(func=mdp.grip_command_obs, params={"command_name": "grip_cmd"})

        gripper_pos = ObsTerm(
            func=mdp.gripper_pos_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[FINGER_JOINT_EXPR])},
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events
##


@configclass
class EventCfg:
    """Reset events executed at each episode boundary."""

    # Randomise the arm joint configuration to expose diverse starts.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    """Dense tracking rewards for EE pose and gripper state."""

    ee_pos_tracking_coarse = RewTerm(
        func=mdp.position_command_error,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME), "command_name": "ee_pose"},
    )
    ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={"std": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME), "command_name": "ee_pose"},
    )

    ee_ori_tracking_coarse = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME), "command_name": "ee_pose"},
    )
    ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=0.5,
        params={"std": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME), "command_name": "ee_pose"},
    )

    grip_tracking = RewTerm(
        func=mdp.gripper_command_tracking,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[FINGER_JOINT_EXPR]), "command_name": "grip_cmd"},
    )

    # Smoothness penalties (ramped up by the curriculum).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot")})


##
# Terminations
##


@configclass
class TerminationsCfg:
    """Episode termination conditions (timeout only)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Curriculum
##


@configclass
class CurriculumCfg:
    """Ramp smoothness penalties to encourage fluid motion over training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.03, "num_steps": 10_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.005, "num_steps": 10_000},
    )


##
# Environment configuration
##


@configclass
class LLEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for the LL EE-tracking policy."""

    scene: LLSceneCfg = LLSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 60 Hz physics, 30 Hz policy (decimation = 2).
        self.decimation = 2
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = tabletop.ROBOT_BASE_ON_TABLE_POS

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Select the joint-position fallback when differential IK is disabled.
        if not USE_DIFFERENTIAL_IK:
            self.actions = JointPositionActionsCfg()
