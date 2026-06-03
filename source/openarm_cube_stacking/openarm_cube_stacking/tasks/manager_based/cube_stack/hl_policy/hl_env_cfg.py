from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..ll_policy.ll_env_cfg import LLSceneCfg, OpenArmCubeStackLLEnvCfg
from . import mdp


@configclass
class HLSceneCfg(LLSceneCfg):
    """Tabletop OpenArm scene used by the classical stack planner."""


@configclass
class HLCommandsCfg:
    """Planner commands with the same names as the LL policy commands."""

    ee_pose = mdp.HLPoseCommandCfg(debug_vis=True, enable_log=False, eval_mode=False)
    grip_cmd = mdp.HLGripCommandCfg()


@configclass
class HLEventCfg:
    """HL play reset: deterministic tabletop layout plus small optional stack/cube jitter."""

    reset_hl_scene = EventTerm(
        func=mdp.reset_hl_scene,
        mode="reset",
        params={
            "cube_position_noise": 0.0,
            "stack_position_noise": 0.0,
            "pose_cmd_name": "ee_pose",
        },
    )


@configclass
class HLTerminationsCfg:
    """HL termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    full_stack = DoneTerm(func=mdp.planner_succeeded)
    planner_failed = DoneTerm(func=mdp.planner_failed)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    stack_collapsed = DoneTerm(func=mdp.stack_collapsed)


@configclass
class OpenArmCubeStackHLEnvCfg(OpenArmCubeStackLLEnvCfg):
    """HL classical stack planner environment driven through the frozen LL policy."""

    scene: HLSceneCfg = HLSceneCfg(num_envs=4, env_spacing=2.5)
    commands: HLCommandsCfg = HLCommandsCfg()
    events: HLEventCfg = HLEventCfg()
    terminations: HLTerminationsCfg = HLTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 45.0
        self.scene.num_envs = 4
        self.scene.ee_frame.debug_vis = True
        self.observations.policy.enable_corruption = False
