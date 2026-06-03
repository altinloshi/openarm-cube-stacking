"""RSL-RL PPO config for the HL environments.

The HL classical-play / eval environments run the frozen LL policy, so they use
the LL PPO runner config (identical network + ``experiment_name`` so the LL
checkpoint loads correctly). Re-exported here for a symmetric package layout.
"""

from ...ll_policy.agents.rsl_rl_ppo_cfg import OpenArmLLPPORunnerCfg  # noqa: F401
