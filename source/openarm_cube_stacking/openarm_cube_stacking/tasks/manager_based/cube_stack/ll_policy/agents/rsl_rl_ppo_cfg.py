from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class OpenArmLLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner for the OpenArm LL goal-conditioned EE-tracking policy.

    The task is reactive (full Markov state is provided), so a compact MLP is
    sufficient. The same runner config is reused by the HL classical-play and
    eval environments so the frozen LL checkpoint loads with a matching network.
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "openarm_ll_ee_tracking"
    run_name = ""
    resume = True
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
