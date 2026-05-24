"""Zero-action agent for the OpenArm cube-stack environments.

Useful for sanity-checking the scene, observations and reward dynamics before
training.

Usage::

    python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action agent for the OpenArm cube-stack env.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-v0", help="Gym env id.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of env steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm_cube_stacking.tasks  # noqa: F401


def main() -> None:
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    print(f"[INFO] Created env '{args_cli.task}' with {args_cli.num_envs} parallel envs.")

    obs, _ = env.reset()
    device = env.unwrapped.device
    action_shape = env.action_space.shape
    zeros = torch.zeros(action_shape, device=device, dtype=torch.float32)

    for step in range(args_cli.num_steps):
        if simulation_app.is_running() is False:
            break
        obs, reward, terminated, truncated, info = env.step(zeros)
        if step % 50 == 0:
            print(f"step {step:5d} | mean reward = {reward.float().mean().item():.4f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
