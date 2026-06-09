# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play variant for Nepher-OpenArm-CubeStack-EndToEnd-Play-v0."""

from isaaclab.utils import configclass

from .cube_stack_env_cfg import OpenArmCubeStackEndToEndEnvCfg


@configclass
class OpenArmCubeStackEndToEndEnvCfg_PLAY(OpenArmCubeStackEndToEndEnvCfg):
    """Deterministic play variant with fewer environments."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_stack_target.params["position_noise"] = 0.0
        self.events.reset_cubes.params["position_noise"] = 0.0
