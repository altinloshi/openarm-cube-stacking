"""HL classical cube-stacking policy for OpenArm.

The High-Level classical stack planner generates EE waypoints
(x, y, z, quat, grip, cube_index) that the frozen LL EE-tracking policy executes,
mounting the OpenArm on the tabletop and stacking five cubes.

Registered environment:
  Nepher-OpenArm-CubeStack-HL-Classical-Play-v0  — runs the planner + frozen LL
      policy on the randomised tabletop stacking scene.

(The deterministic tournament-evaluation environment
``Nepher-OpenArm-CubeStack-Eval-v0`` is registered in the ``eval`` package.)
"""

import gymnasium as gym

from ..ll_policy import agents as ll_agents

gym.register(
    id="Nepher-OpenArm-CubeStack-HL-Classical-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg_play:HLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{ll_agents.__name__}.rsl_rl_ppo_cfg:OpenArmLLPPORunnerCfg",
    },
)
