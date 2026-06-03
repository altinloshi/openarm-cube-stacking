"""End-to-end cube stacking sub-package.

Re-exports the existing end-to-end env configs so that the new directory layout
is consistent without duplicating code.  The actual implementation lives in the
parent cube_stack package.
"""

import gymnasium as gym

from . import agents
from .cube_stack_env_cfg import OpenArmCubeStackEndToEndEnvCfg
from .cube_stack_env_cfg_play import OpenArmCubeStackEndToEndEnvCfg_PLAY

gym.register(
    id="Nepher-OpenArm-CubeStack-EndToEnd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg:OpenArmCubeStackEndToEndEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackEndToEndPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-EndToEnd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEndToEndEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackEndToEndPPORunnerCfg",
    },
    disable_env_checker=True,
)
