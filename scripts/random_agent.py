"""Random-action agent for the OpenArm cube-stack environments.

Usage::

    python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Random-action agent for the OpenArm cube-stack env.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-v0", help="Gym env id.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of env steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app (must happen before any isaaclab/gym env imports).
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------ rest after sim launch ------
import gymnasium as gym
import torch

import openarm_cube_stacking.tasks  # noqa: F401  (registers envs)


def main() -> None:
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    print(f"[INFO] Created env '{args_cli.task}' with {args_cli.num_envs} parallel envs.")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space:      {env.action_space}")

    obs, _ = env.reset()
    action_space = env.action_space
    for step in range(args_cli.num_steps):
        if simulation_app.is_running() is False:
            break
        actions = torch.from_numpy(action_space.sample()).to(env.unwrapped.device).float()
        obs, reward, terminated, truncated, info = env.step(actions)
        if step % 50 == 0:
            print(f"step {step:5d} | mean reward = {reward.float().mean().item():.4f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
