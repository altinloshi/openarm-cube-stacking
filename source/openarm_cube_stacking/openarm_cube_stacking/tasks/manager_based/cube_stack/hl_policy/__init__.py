"""High-level classical OpenArm cube-stacking policy registrations."""

import gymnasium as gym

from ..ll_policy import agents

gym.register(
    id="Nepher-OpenArm-CubeStack-HL-Classical-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg_play:OpenArmCubeStackHLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackLLPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg_eval:OpenArmCubeStackHLEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackLLPPORunnerCfg",
    },
    disable_env_checker=True,
)
