"""Play / evaluation configuration for ``Nepher-OpenArm-CubeStack-Play-v0``.

Inherits from :class:`OpenArmCubeStackEnvCfg`; adjusts the scene size and disables
observation noise so that pretrained policies can be replayed deterministically.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .cube_stack_env_cfg import OpenArmCubeStackEnvCfg


@configclass
class OpenArmCubeStackEnvCfg_PLAY(OpenArmCubeStackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Deterministic rollout.
        self.observations.policy.enable_corruption = False
