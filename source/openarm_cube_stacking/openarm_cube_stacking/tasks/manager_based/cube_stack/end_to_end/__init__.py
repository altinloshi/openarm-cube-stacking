"""End-to-end OpenArm cube-stacking baseline task.

This is the original monolithic task: a single RL policy learns the full
five-cube stacking behaviour directly from joint-position + binary-gripper
actions. It is preserved unchanged as a baseline beside the new hierarchical
(LL + HL classical planner) tournament pipeline.

Registered environments:
  Nepher-OpenArm-CubeStack-EndToEnd-v0        — training
  Nepher-OpenArm-CubeStack-EndToEnd-Play-v0   — play / evaluation
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-OpenArm-CubeStack-EndToEnd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg:OpenArmCubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-EndToEnd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)
