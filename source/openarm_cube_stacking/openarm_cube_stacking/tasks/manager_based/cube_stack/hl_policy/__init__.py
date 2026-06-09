# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HL policy sub-package: classical planner + LL policy executor."""

import gymnasium as gym

from . import agents
from .hl_env_cfg import OpenArmHLEnvCfg
from .hl_env_cfg_eval import OpenArmEvalEnvCfg
from .hl_env_cfg_play import OpenArmHLEnvCfg_PLAY

gym.register(
    id="Nepher-OpenArm-CubeStack-HL-Classical-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg_play:OpenArmHLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmHLRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg_eval:OpenArmEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmHLRunnerCfg",
    },
    disable_env_checker=True,
)
