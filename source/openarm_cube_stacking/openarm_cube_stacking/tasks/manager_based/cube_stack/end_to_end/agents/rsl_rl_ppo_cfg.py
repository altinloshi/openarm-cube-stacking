# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the end-to-end cube stacking task."""

from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import OpenArmCubeStackPPORunnerCfg


@configclass
class OpenArmCubeStackEndToEndPPORunnerCfg(OpenArmCubeStackPPORunnerCfg):
    """Runner config for Nepher-OpenArm-CubeStack-EndToEnd-v0."""

    experiment_name = "openarm_cube_stack_end_to_end"
