# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner config for HL environments.

The HL play/eval environments load the FROZEN LL policy, so the
experiment_name points to the LL training logs to find the checkpoint.
"""

from isaaclab.utils import configclass

from ...ll_policy.agents.rsl_rl_ppo_cfg import OpenArmLLPPORunnerCfg


@configclass
class OpenArmHLRunnerCfg(OpenArmLLPPORunnerCfg):
    """Runner config for HL-Classical-Play and Eval environments.

    Points to the LL policy experiment so play.py loads the trained LL weights.
    At play time, the environment uses the classical planner command term, and
    the LL policy executes the EE tracking.
    """

    experiment_name = "openarm_cube_stack_ll"
    # Load latest run/checkpoint automatically
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
