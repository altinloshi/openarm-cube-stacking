"""RSL-RL PPO runner configuration for the 5-cube stacking task."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OpenArmCubeStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for Nepher-OpenArm-CubeStack-v0.

    Hyperparameters are tuned for a manipulation task with a moderately long
    episode (20 s) and sparse–dense mixed reward structure.  Start with these
    defaults and adjust based on training curves.
    """

    # -----------------------------------------------------------------------
    # Runner settings
    # -----------------------------------------------------------------------
    num_steps_per_env: int = 32
    max_iterations: int = 5000
    save_interval: int = 100
    experiment_name: str = "openarm_cube_stack"
    empirical_normalization: bool = False

    # -----------------------------------------------------------------------
    # Policy (actor-critic) architecture
    # -----------------------------------------------------------------------
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # -----------------------------------------------------------------------
    # PPO algorithm settings
    # -----------------------------------------------------------------------
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
