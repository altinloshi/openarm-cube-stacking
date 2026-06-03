from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from . import mdp
from .hl_env_cfg import HLEventCfg, OpenArmCubeStackHLEnvCfg


@configclass
class HLEvalEventCfg(HLEventCfg):
    """Deterministic tournament reset from the 30 scenario table."""

    reset_hl_scene = EventTerm(
        func=mdp.reset_hl_scene_from_scenarios,
        mode="reset",
        params={"pose_cmd_name": "ee_pose"},
    )


@configclass
class OpenArmCubeStackHLEnvCfg_EVAL(OpenArmCubeStackHLEnvCfg):
    """Deterministic 30-scenario evaluation config."""

    events: HLEvalEventCfg = HLEvalEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 30
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
        self.commands.ee_pose.enable_log = False
        self.commands.ee_pose.eval_mode = True
        self.seed = 2026
