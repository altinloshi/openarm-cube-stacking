"""RSL-RL PPO configuration for the OpenArm cube-stack task.

Mirrors the structure of the official Isaac Lab manipulation PPO configs and
provides a sensible starting point. Tune ``max_iterations`` and the network
sizes for your specific compute budget.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OpenArmCubeStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for ``Nepher-OpenArm-CubeStack-v0``."""

    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 50
    experiment_name = "openarm_cube_stack"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
