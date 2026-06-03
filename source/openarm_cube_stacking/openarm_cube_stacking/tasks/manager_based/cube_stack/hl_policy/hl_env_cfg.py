"""High-level environment base config for HL play and eval.

This environment has:
- OpenArm fixed in the official lift-cube scene layout
- Five DexCube objects near the standard lift-cube spawn area
- ClassicalStackPlannerCommand providing EE target poses
- LL-compatible observation space (so a frozen LL policy can be applied)

Usage
-----
The HL-Classical-Play-v0 environment is used with the standard play.py script:

    python scripts/rsl_rl/play.py \\
        --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 \\
        --checkpoint=best_policy/best_policy.pt

The play.py loads the LL policy weights (pointed to by OpenArmHLRunnerCfg)
and runs the LL actor network on the HL env observations.  The classical
planner drives the ee_pose command, so the LL policy naturally executes the
planned trajectory.
"""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..tabletop_scene_cfg import (
    CUBE_NAMES,
    TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS,
    TABLETOP_STACK_BASE_LOCAL_POS,
    OpenArmTabletopWithCubesSceneCfg,
)
from . import mdp


# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLSceneCfg(OpenArmTabletopWithCubesSceneCfg):
    """Official OpenArm lift-style USD table scene with robot and five cubes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.ee_frame.debug_vis = False


# ─────────────────────────────────────────────────────────────────────────────
# Actions (same as LL – LL policy outputs joint positions)
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLActionsCfg:
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


# ─────────────────────────────────────────────────────────────────────────────
# Commands  (classical planner drives the EE target)
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLCommandsCfg:
    """Classical planner command term: outputs EE target pose for LL tracking."""

    ee_pose = mdp.ClassicalStackPlannerCommandCfg(
        debug_vis=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observations (LL-compatible policy group + optional extra groups)
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Same observation structure as LL policy – so frozen LL weights apply."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        current_ee_pos_b = ObsTerm(func=mdp.ee_pos_b)
        current_ee_quat_b = ObsTerm(func=mdp.ee_quat_b)
        target_ee_pose = ObsTerm(
            func=mdp.target_ee_pose_command,
            params={"command_name": "ee_pose"},
        )
        target_grip = ObsTerm(func=mdp.target_gripper_cmd)
        gripper_open = ObsTerm(func=mdp.gripper_opening_norm)
        last_act = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StackStateCfg(ObsGroup):
        """Extra group for monitoring stack progress (not fed to LL policy)."""

        cubes_w = ObsTerm(func=mdp.cube_positions_w, params={"cube_names": list(CUBE_NAMES)})
        planner_stg = ObsTerm(func=mdp.planner_stage)
        planner_cube = ObsTerm(func=mdp.planner_cube_idx)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    stack_state: StackStateCfg = StackStateCfg()


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLEventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reset_stack_target = EventTerm(
        func=mdp.reset_stack_target_lift_style,
        mode="reset",
        params={
            "local_stack_base": TABLETOP_STACK_BASE_LOCAL_POS,
            "position_noise": 0.02,
        },
    )
    reset_cubes = EventTerm(
        func=mdp.reset_cubes_lift_style,
        mode="reset",
        params={
            "cube_names": list(CUBE_NAMES),
            "local_positions": TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS,
            "position_noise": 0.015,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rewards (light-weight monitoring; HL env is not trained directly)
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLRewardsCfg:
    cubes_at_target = RewTerm(
        func=mdp.cubes_at_target,
        weight=1.0,
        params={"threshold": 0.04},
    )
    stack_success = RewTerm(
        func=mdp.stack_success_bonus,
        weight=5.0,
        params={"threshold": 0.04},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Terminations
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class HLTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    planner_done = DoneTerm(func=mdp.planner_done)


# ─────────────────────────────────────────────────────────────────────────────
# Base HL environment config
# ─────────────────────────────────────────────────────────────────────────────


@configclass
class OpenArmHLEnvCfg(ManagerBasedRLEnvCfg):
    """Base config shared by HL-Classical-Play and Eval environments."""

    scene: HLSceneCfg = HLSceneCfg(num_envs=4, env_spacing=2.5)
    observations: HLObservationsCfg = HLObservationsCfg()
    actions: HLActionsCfg = HLActionsCfg()
    commands: HLCommandsCfg = HLCommandsCfg()
    events: HLEventCfg = HLEventCfg()
    rewards: HLRewardsCfg = HLRewardsCfg()
    terminations: HLTerminationsCfg = HLTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 60.0  # longer episodes for full 5-cube stack

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
