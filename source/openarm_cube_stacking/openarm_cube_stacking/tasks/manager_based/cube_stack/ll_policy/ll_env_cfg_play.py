from isaaclab.utils import configclass

from .ll_env_cfg import OpenArmCubeStackLLEnvCfg


@configclass
class OpenArmCubeStackLLEnvCfg_PLAY(OpenArmCubeStackLLEnvCfg):
    """Play config: fewer envs, no observation noise, debug visualization enabled."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.ee_frame.debug_vis = True
        self.commands.ee_pose.debug_vis = True
        self.observations.policy.enable_corruption = False
