# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LL policy sub-package: goal-conditioned EE tracking for OpenArm."""

import gymnasium as gym

from . import agents
from .ll_env_cfg import OpenArmLLEnvCfg
from .ll_env_cfg_play import OpenArmLLEnvCfg_PLAY

gym.register(
    id="Nepher-OpenArm-CubeStack-LL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:OpenArmLLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLLPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-LL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg_play:OpenArmLLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLLPPORunnerCfg",
    },
    disable_env_checker=True,
)
