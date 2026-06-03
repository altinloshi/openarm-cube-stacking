"""Low-level goal-conditioned OpenArm EE tracking policy registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-OpenArm-CubeStack-LL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:OpenArmCubeStackLLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackLLPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-LL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg_play:OpenArmCubeStackLLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackLLPPORunnerCfg",
    },
    disable_env_checker=True,
)
