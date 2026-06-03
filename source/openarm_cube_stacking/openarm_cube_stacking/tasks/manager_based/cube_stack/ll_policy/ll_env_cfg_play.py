from __future__ import annotations

"""Play / evaluation configuration for the OpenArm LL EE-tracking policy.

Fewer environments, no observation noise, and the end-effector target debug
frame enabled. Used by ``Nepher-OpenArm-CubeStack-LL-Play-v0``.
"""

from isaaclab.utils import configclass

from .ll_env_cfg import LLEnvCfg


@configclass
class LLEnvCfg_PLAY(LLEnvCfg):
    """Play-time variant of the LL EE-tracking environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Show the EE target debug frame during play.
        self.scene.ee_frame.debug_vis = True
