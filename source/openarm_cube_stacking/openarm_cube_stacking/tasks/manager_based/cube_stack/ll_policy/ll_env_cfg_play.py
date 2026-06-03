"""Play variant for Nepher-OpenArm-CubeStack-LL-Play-v0."""

from isaaclab.utils import configclass

from .ll_env_cfg import OpenArmLLEnvCfg


@configclass
class OpenArmLLEnvCfg_PLAY(OpenArmLLEnvCfg):
    """Play-time LL env: fewer envs, no noise, debug visualisation on."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        # Disable observation noise for clean visualisation
        self.observations.policy.enable_corruption = False
        # Show EE target frame marker
        self.scene.ee_frame.debug_vis = True
        # Show command debug marker
        self.commands.ee_pose.debug_vis = True
