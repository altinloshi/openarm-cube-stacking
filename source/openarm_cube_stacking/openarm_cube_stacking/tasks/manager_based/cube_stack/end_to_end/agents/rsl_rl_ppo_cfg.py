"""RSL-RL PPO config for the end-to-end cube stacking task."""

from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import OpenArmCubeStackPPORunnerCfg


@configclass
class OpenArmCubeStackEndToEndPPORunnerCfg(OpenArmCubeStackPPORunnerCfg):
    """Runner config for Nepher-OpenArm-CubeStack-EndToEnd-v0."""

    experiment_name = "openarm_cube_stack_end_to_end"
