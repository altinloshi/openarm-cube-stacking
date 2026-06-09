# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner config for the LL policy EE tracking task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OpenArmLLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for Nepher-OpenArm-CubeStack-LL-v0.

    The LL policy is a relatively simple goal-conditioned EE tracker;
    a smaller network and fewer iterations are needed compared to the
    full stacking task.
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "openarm_cube_stack_ll"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
