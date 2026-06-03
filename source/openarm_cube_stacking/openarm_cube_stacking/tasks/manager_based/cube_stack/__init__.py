"""OpenArm cube-stacking tasks.

Two pipelines live side by side under this package:

* ``end_to_end`` — the original monolithic baseline task (one policy learns the
  whole five-cube stack). Kept as a baseline.
* ``ll_policy`` / ``hl_policy`` / ``eval`` — the tournament-ready hierarchical
  pipeline that mirrors the Franka-Mani-Base architecture:

      HL classical stack planner
          -> sends waypoints (x, y, z, quat, grip, cube_index)
      LL goal-conditioned EE-tracking policy
          -> tracks the target EE pose + binary gripper command
      OpenArm mounted on the tabletop

Registered environments
------------------------
Legacy (kept working, alias the end-to-end baseline):
  Nepher-OpenArm-CubeStack-v0
  Nepher-OpenArm-CubeStack-Play-v0

End-to-end baseline:
  Nepher-OpenArm-CubeStack-EndToEnd-v0
  Nepher-OpenArm-CubeStack-EndToEnd-Play-v0

Hierarchical pipeline:
  Nepher-OpenArm-CubeStack-LL-v0
  Nepher-OpenArm-CubeStack-LL-Play-v0
  Nepher-OpenArm-CubeStack-HL-Classical-Play-v0
  Nepher-OpenArm-CubeStack-Eval-v0
"""

import gymnasium as gym

from . import end_to_end  # noqa: F401  (registers EndToEnd envs)
from . import ll_policy  # noqa: F401  (registers LL envs)
from . import hl_policy  # noqa: F401  (registers HL envs)
from . import eval as eval_pipeline  # noqa: F401  (registers Eval env)
from .end_to_end import agents as _e2e_agents

# ---------------------------------------------------------------------------
# Legacy environment IDs — alias the end-to-end baseline so existing commands
# (``--task=Nepher-OpenArm-CubeStack-v0`` / ``-Play-v0``) keep working.
# ---------------------------------------------------------------------------

gym.register(
    id="Nepher-OpenArm-CubeStack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{end_to_end.__name__}.cube_stack_env_cfg:OpenArmCubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_e2e_agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Nepher-OpenArm-CubeStack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{end_to_end.__name__}.cube_stack_env_cfg_play:OpenArmCubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_e2e_agents.__name__}.rsl_rl_ppo_cfg:OpenArmCubeStackPPORunnerCfg",
    },
    disable_env_checker=True,
)
