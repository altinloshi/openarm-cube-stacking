import gymnasium as gym

from . import agents
from . import hl_policy, ll_policy  # noqa: F401
from .end_to_end import agents as end_to_end_agents


gym.register(
    id="Nepher-OpenArm-CubeStack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg:OpenArmCubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-EndToEnd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.end_to_end.cube_stack_env_cfg:OpenArmCubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{end_to_end_agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-EndToEnd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.end_to_end.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{end_to_end_agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

