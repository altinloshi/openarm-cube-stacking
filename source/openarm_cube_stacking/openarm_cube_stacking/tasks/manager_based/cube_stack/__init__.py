"""OpenArm 5-cube sequential stacking task.

This module registers two Gym environments:

* ``Nepher-OpenArm-CubeStack-v0``      – training configuration (``OpenArmCubeStackEnvCfg``)
* ``Nepher-OpenArm-CubeStack-Play-v0`` – evaluation / play configuration (``OpenArmCubeStackEnvCfg_PLAY``)

Both environments use the manager-based :class:`isaaclab.envs.ManagerBasedRLEnv`
and share the same RSL-RL PPO configuration.
"""

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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-OpenArm-CubeStack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
)
