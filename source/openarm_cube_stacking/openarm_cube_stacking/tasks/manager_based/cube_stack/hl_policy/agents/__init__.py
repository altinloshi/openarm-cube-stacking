"""Agent configurations for the HL classical cube-stacking environment.

The HL classical-play and eval environments are executed by the **frozen LL
policy**, so they reuse the LL PPO runner config (matching network + experiment
name) re-exported here.
"""

from ...ll_policy.agents.rsl_rl_ppo_cfg import OpenArmLLPPORunnerCfg  # noqa: F401
