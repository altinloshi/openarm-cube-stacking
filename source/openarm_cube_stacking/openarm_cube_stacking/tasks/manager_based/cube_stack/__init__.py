"""Gym registration for the OpenArm cube stacking task."""

from __future__ import annotations

import gymnasium as gym

from . import agents


def _register_envs() -> None:
    train_id = "Nepher-OpenArm-CubeStack-v0"
    play_id = "Nepher-OpenArm-CubeStack-Play-v0"

    if train_id not in gym.registry:
        gym.register(
            id=train_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg:OpenArmCubeStackEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
            },
        )

    if play_id not in gym.registry:
        gym.register(
            id=play_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
            },
        )


_register_envs()

__all__ = ["agents"]
