"""Cube stacking task package.

Registers all Gymnasium environments for:
  - Existing end-to-end baseline (Nepher-OpenArm-CubeStack-v0 / -Play-v0)
  - New end-to-end aliases (-EndToEnd-v0 / -EndToEnd-Play-v0)
  - Low-level EE tracker (-LL-v0 / -LL-Play-v0)
  - High-level classical planner play (-HL-Classical-Play-v0)
  - Deterministic tournament evaluation (-Eval-v0)
"""

import gymnasium as gym

from . import agents

# ── Existing end-to-end environments (backward-compatible) ───────────────────

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
    id="Nepher-OpenArm-CubeStack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

# ── New sub-package environments ─────────────────────────────────────────────
# Each sub-package __init__.py registers its own environments via gym.register.

from . import end_to_end  # noqa: F401, E402  (registers EndToEnd-v0 / -Play-v0)
from . import ll_policy   # noqa: F401, E402  (registers LL-v0 / LL-Play-v0)
from . import hl_policy   # noqa: F401, E402  (registers HL-Classical-Play-v0 / Eval-v0)

