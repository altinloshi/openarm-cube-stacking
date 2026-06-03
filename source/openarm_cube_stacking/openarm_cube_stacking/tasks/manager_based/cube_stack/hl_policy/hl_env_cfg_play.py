from __future__ import annotations

"""Play configuration for the HL classical cube-stacking environment.

A handful of environments, randomised (but reachable) cube spawns, planner
logging and the planner-target debug frame enabled. Used by
``Nepher-OpenArm-CubeStack-HL-Classical-Play-v0``.
"""

from isaaclab.utils import configclass

from .hl_env_cfg import HLEnvCfg


@configclass
class HLEnvCfg_PLAY(HLEnvCfg):
    """Play-time variant of the HL classical stacking environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.commands.ee_pose.enable_log = True
        self.commands.ee_pose.debug_vis = True
        self.scene.ee_frame.debug_vis = True
