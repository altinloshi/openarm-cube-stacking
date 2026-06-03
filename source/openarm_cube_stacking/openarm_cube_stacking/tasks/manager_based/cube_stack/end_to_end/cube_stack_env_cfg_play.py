from isaaclab.utils import configclass

from .cube_stack_env_cfg import OpenArmCubeStackEnvCfg


@configclass
class OpenArmCubeStackEnvCfg_PLAY(OpenArmCubeStackEnvCfg):
    """Play-time variant with fewer environments and deterministic observations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_stack_target.params["position_noise"] = 0.0
        self.events.reset_cubes.params["position_noise"] = 0.0

