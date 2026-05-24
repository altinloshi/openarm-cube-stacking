"""Play a trained PPO policy in OpenArm cube stacking environment."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-Play-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run loop with real-time sleep.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import openarm_cube_stacking.tasks  # noqa: F401


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    train_task_name = args_cli.task.replace("-Play", "")
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = args_cli.checkpoint or get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint for {train_task_name}: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)
        sleep_time = dt - (time.time() - start)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
