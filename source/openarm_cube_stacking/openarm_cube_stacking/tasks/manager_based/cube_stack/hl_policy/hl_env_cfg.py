from __future__ import annotations

"""High-Level classical cube-stacking environment for OpenArm.

The HL env extends the LL EE-tracking environment:

1. **Scene** — adds five cubes on the tabletop (``OpenArmTabletopStackSceneCfg``).
2. **Commands** — replaces the random ``UniformPoseCommand`` / ``GripperCommand``
   pair with ``HLStackPoseCommand`` + ``HLGripCommand``, driven by the
   ``ClassicalStackPlanner`` state machine. The LL observation layout is
   structurally identical to training, so the frozen LL checkpoint runs unchanged.
3. **Events / Terminations** — randomises cube spawns + stack base, and
   terminates on full-stack success, stack failure, or timeout.

The OpenArm is mounted on the tabletop and the planner-target debug frame is
visualised. Run with::

    python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import tabletop
from ..ll_policy.ll_env_cfg import LLEnvCfg
from ..ll_policy.ll_env_cfg import RewardsCfg as LLRewardsCfg
from ..tabletop import OpenArmTabletopStackSceneCfg
from . import mdp


##
# Scene
##


@configclass
class HLSceneCfg(OpenArmTabletopStackSceneCfg):
    """Tabletop scene with five cubes and the OpenArm mounted on the table.

    The robot variant is selected in ``LLEnvCfg.__post_init__`` (inherited).
    """


##
# Commands
##


@configclass
class HLCommandsCfg:
    """Planner-driven EE pose command + mirrored gripper command.

    ``ee_pose`` must be declared before ``grip_cmd`` so the planner pose updates
    before the gripper command mirrors its grip value.
    """

    ee_pose: mdp.HLStackPoseCommandCfg = mdp.HLStackPoseCommandCfg(
        cube_names=list(tabletop.CUBE_NAMES),
        stack_base_local=tabletop.DEFAULT_STACK_BASE_LOCAL_POS,
        debug_vis=True,
        enable_log=True,
    )

    grip_cmd: mdp.HLGripCommandCfg = mdp.HLGripCommandCfg()


##
# Events
##


@configclass
class HLEventCfg:
    """Randomised reset events for HL classical play.

    Order matters: reset the robot first, then place cubes and the stack base.
    """

    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_cubes_and_stack = EventTerm(
        func=mdp.reset_cubes_and_stack,
        mode="reset",
        params={
            "cube_names": list(tabletop.CUBE_NAMES),
            "local_positions": tabletop.DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
            "stack_base_local": tabletop.DEFAULT_STACK_BASE_LOCAL_POS,
            "cube_position_noise": 0.02,
            "stack_position_noise": 0.0,
            "pose_cmd_name": "ee_pose",
        },
    )


##
# Rewards (LL tracking rewards + a diagnostic stack-progress term)
##


@configclass
class HLRewardsCfg(LLRewardsCfg):
    """Inherits the LL tracking rewards (so the LL curriculum stays valid) and
    adds a zero-weight diagnostic stack-progress term."""

    stack_progress = RewTerm(func=mdp.stack_progress, weight=0.0)


##
# Terminations
##


@configclass
class HLTerminationsCfg:
    """Success / failure / timeout terminations for cube stacking."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    stack_complete = DoneTerm(
        func=mdp.all_cubes_stacked,
        params={"cube_names": list(tabletop.CUBE_NAMES), "pose_cmd_name": "ee_pose"},
    )

    stack_failed = DoneTerm(
        func=mdp.stack_failed,
        params={"cube_names": list(tabletop.CUBE_NAMES), "pose_cmd_name": "ee_pose"},
    )


##
# Environment configuration
##


@configclass
class HLEnvCfg(LLEnvCfg):
    """HL classical-play configuration (reuses LL actions / observations)."""

    scene: HLSceneCfg = HLSceneCfg(num_envs=4, env_spacing=2.5)
    commands: HLCommandsCfg = HLCommandsCfg()
    events: HLEventCfg = HLEventCfg()
    rewards: HLRewardsCfg = HLRewardsCfg()
    terminations: HLTerminationsCfg = HLTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Sequential five-cube stacking takes much longer than pure EE tracking.
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        # Show the EE target debug frame during play.
        self.scene.ee_frame.debug_vis = True
