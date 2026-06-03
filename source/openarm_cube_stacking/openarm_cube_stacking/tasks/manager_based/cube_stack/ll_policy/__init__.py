"""Low-Level goal-conditioned EE-tracking policy for OpenArm.

Trains a reactive policy that drives the end-effector (``openarm_hand``) to any
commanded target pose (x, y, z, rotx, roty, rotz) and matches a commanded binary
gripper state, from an arbitrary arm configuration. It does not learn cube
stacking; the High-Level classical planner composes these tracked waypoints into
a full stack.

Registered environments:
  Nepher-OpenArm-CubeStack-LL-v0        — training  (4096 envs, noise on)
  Nepher-OpenArm-CubeStack-LL-Play-v0   — play      (32 envs, noise off, debug vis)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-OpenArm-CubeStack-LL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLLPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-OpenArm-CubeStack-LL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg_play:LLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLLPPORunnerCfg",
    },
)
