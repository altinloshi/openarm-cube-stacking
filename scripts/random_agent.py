"""Run random actions in a registered Isaac Lab environment."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run random actions in OpenArm cube stacking env.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of vectorized environments.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum environment steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm_cube_stacking.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.max_steps:
        actions = 2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0
        env.step(actions)
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
