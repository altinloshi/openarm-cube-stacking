"""Deterministic tournament evaluation for OpenArm cube stacking.

Registers the evaluation environment and exposes the deterministic scenarios and
metric helpers used by ``scripts/eval/evaluate_stack.py``.

Registered environment:
  Nepher-OpenArm-CubeStack-Eval-v0 — 30 deterministic scenarios, no noise,
      executed by the frozen LL policy + classical stack planner.
"""

import gymnasium as gym

from ..ll_policy import agents as ll_agents
from . import metrics, scenarios  # noqa: F401

gym.register(
    id="Nepher-OpenArm-CubeStack-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "openarm_cube_stacking.tasks.manager_based.cube_stack.hl_policy.hl_env_cfg_eval:HLEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{ll_agents.__name__}.rsl_rl_ppo_cfg:OpenArmLLPPORunnerCfg",
    },
)
