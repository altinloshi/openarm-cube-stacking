"""Export the OpenArm LL policy from best_policy/best_policy.pt."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export OpenArm LL policy to TorchScript and ONNX.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-LL-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import openarm_cube_stacking.tasks  # noqa: F401, E402
import policy_paths  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resume_path = policy_paths.sync_best_policy(
        agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint, explicit_checkpoint=args_cli.checkpoint
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    os.makedirs(policy_paths.BEST_POLICY_EXPORT_DIR, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=policy_paths.BEST_POLICY_EXPORT_DIR, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=policy_paths.BEST_POLICY_EXPORT_DIR, filename="policy.onnx")
    print(f"[INFO] Exported policy to: {policy_paths.BEST_POLICY_EXPORT_DIR}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
