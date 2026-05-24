"""Gym environment registration for the OpenArm 5-cube stacking task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Nepher-OpenArm-CubeStack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg:OpenArmCubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Nepher-OpenArm-CubeStack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg"
        ),
    },
)
