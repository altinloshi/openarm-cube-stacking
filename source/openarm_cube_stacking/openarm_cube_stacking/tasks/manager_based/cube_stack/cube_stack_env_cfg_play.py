"""Play configuration for OpenArm cube stacking."""

from isaaclab.utils import configclass

from .cube_stack_env_cfg import OpenArmCubeStackEnvCfg


@configclass
class OpenArmCubeStackEnvCfg_PLAY(OpenArmCubeStackEnvCfg):
    """Smaller, less-randomized scene for policy playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_stack_target.params["xy_noise"] = 0.0
        self.events.reset_cubes.params["xy_noise"] = 0.0
