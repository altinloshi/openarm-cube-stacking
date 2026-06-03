"""Low-level policy environment config for Nepher-OpenArm-CubeStack-LL-v0.

Purpose
-------
Train a goal-conditioned end-effector tracking controller.  The LL policy
learns to move the OpenArm EE to a commanded pose and to open/close the
gripper on command.  It does NOT learn full cube stacking; that is the
HL task's responsibility.

Action space
------------
  arm_action   : JointPositionActionCfg covering openarm_joint[1-7], 7 dims
  gripper_action: BinaryJointPositionActionCfg for openarm_finger_joint.*, 1 dim
  Total: 8 dims

Observation space (concatenated, ~42 dims)
------------------------------------------
  joint_pos_rel          : n_arm + n_finger joints
  joint_vel_rel          : same
  ee_pos_b               : 3 (EE position in robot base frame)
  ee_quat_b              : 4 (EE orientation in robot base frame, wxyz)
  target_ee_pose_command : 7 (pos(3) + quat(4) from command manager)
  target_gripper_cmd     : 1 (0=close, 1=open)
  gripper_opening_norm   : 1 (current opening in [0, 1])
  last_action            : 8

To switch to DifferentialIK control, replace arm_action with:
  DifferentialInverseKinematicsActionCfg(
      asset_name="robot",
      joint_names=["openarm_joint[1-7]"],
      body_name="openarm_hand",
      controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
  )
and update the action_dim accordingly.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..tabletop_scene_cfg import OpenArmTabletopSceneCfg
from . import mdp


# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLSceneCfg(OpenArmTabletopSceneCfg):
    """LL task scene: official OpenArm lift layout with no cubes.

    No cubes: the LL policy only needs the arm, standard table, and EE frame.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # Enable EE debug visualisation during play
        self.ee_frame.debug_vis = False


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLActionsCfg:
    """OpenArm joint-position arm control + binary finger control."""

    # 7-DOF arm joints, position control with 0.5 scale
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )

    # Binary gripper: 1 = open (0.044 m), 0 = close (0.0 m)
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLCommandsCfg:
    """EE pose command in the robot base frame for the lift-style workspace."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        # openarm_hand is the EE rigid body; openarm_ee_tcp is a site/frame.
        # Using openarm_hand here for UniformPoseCommandCfg which needs a rigid body.
        body_name="openarm_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Covers the official lift cube spawn area and the stack target at x=0.55.
            pos_x=(0.20, 0.60),
            pos_y=(-0.25, 0.25),
            pos_z=(0.05, 0.40),
            # Mostly pointing downward with some roll/yaw variation
            roll=(-math.pi / 8, math.pi / 8),
            pitch=(math.pi * 0.7, math.pi),
            yaw=(-math.pi / 4, math.pi / 4),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observations
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLObservationsCfg:
    """Observations for the LL policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Current EE pose in robot base frame
        current_ee_pos_b = ObsTerm(func=mdp.ee_pos_b)
        current_ee_quat_b = ObsTerm(func=mdp.ee_quat_b)
        # Command targets
        target_ee_pose = ObsTerm(
            func=mdp.target_ee_pose_command,
            params={"command_name": "ee_pose"},
        )
        target_grip = ObsTerm(func=mdp.target_gripper_cmd)
        # Current gripper state
        gripper_open = ObsTerm(func=mdp.gripper_opening_norm)
        # History
        last_act = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLEventCfg:
    """Reset events for the LL task."""

    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Sample binary gripper command once per episode
    reset_gripper_cmd = EventTerm(
        func=mdp.reset_gripper_command,
        mode="reset",
        params={"open_probability": 0.5},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rewards
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLRewardsCfg:
    """Reward terms for EE pose tracking and gripper command following.

    Curriculum: action_rate and joint_vel penalties start near zero and
    increase linearly.  Apply ramp via weight_curriculum in training config.
    """

    # EE position tracking
    ee_pos_coarse = RewTerm(
        func=mdp.ee_position_tracking_coarse,
        weight=1.0,
        params={"std": 0.20, "command_name": "ee_pose"},
    )
    ee_pos_fine = RewTerm(
        func=mdp.ee_position_tracking_fine,
        weight=2.0,
        params={"std": 0.05, "command_name": "ee_pose"},
    )

    # EE orientation tracking
    ee_ori_coarse = RewTerm(
        func=mdp.ee_orientation_tracking_coarse,
        weight=0.5,
        params={"std": 0.5, "command_name": "ee_pose"},
    )
    ee_ori_fine = RewTerm(
        func=mdp.ee_orientation_tracking_fine,
        weight=1.0,
        params={"std": 0.15, "command_name": "ee_pose"},
    )

    # Gripper command tracking
    gripper_tracking = RewTerm(
        func=mdp.gripper_command_tracking,
        weight=0.5,
    )

    # Regularisation (small initial weights; increase with curriculum)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1.0e-3,
    )
    joint_vel = RewTerm(
        func=mdp.joint_velocity_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Terminations
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class LLTerminationsCfg:
    """Terminations: timeout only."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main environment config
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class OpenArmLLEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL config for ``Nepher-OpenArm-CubeStack-LL-v0``."""

    scene: LLSceneCfg = LLSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: LLObservationsCfg = LLObservationsCfg()
    actions: LLActionsCfg = LLActionsCfg()
    commands: LLCommandsCfg = LLCommandsCfg()
    events: LLEventCfg = LLEventCfg()
    rewards: LLRewardsCfg = LLRewardsCfg()
    terminations: LLTerminationsCfg = LLTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 8.0  # short episodes for LL tracking

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
