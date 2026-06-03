from isaaclab.utils import configclass

from .hl_env_cfg import OpenArmCubeStackHLEnvCfg


@configclass
class OpenArmCubeStackHLEnvCfg_PLAY(OpenArmCubeStackHLEnvCfg):
    """Play config for the classical five-cube stack planner."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.scene.ee_frame.debug_vis = True
        self.commands.ee_pose.debug_vis = True
        self.commands.ee_pose.enable_log = True
        self.commands.ee_pose.log_interval = 120
