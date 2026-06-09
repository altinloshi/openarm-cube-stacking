# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play variant for Nepher-OpenArm-CubeStack-HL-Classical-Play-v0."""

from isaaclab.utils import configclass

from .hl_env_cfg import OpenArmHLEnvCfg


@configclass
class OpenArmHLEnvCfg_PLAY(OpenArmHLEnvCfg):
    """Play-time HL env: debug visualisation enabled, small env count."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # No observation noise during play
        self.observations.policy.enable_corruption = False
        # Enable EE target frame debug marker
        self.scene.ee_frame.debug_vis = True
        # Show planner target waypoint in the scene
        self.commands.ee_pose.debug_vis = True
