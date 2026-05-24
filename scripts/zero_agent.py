"""Run zero actions in the OpenArm cube stacking environment.

Usage::

    python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4

Sends all-zero actions to verify that the environment remains stable with no
policy input (i.e., the robot holds its default pose and cubes stay on the
table).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Zero agent for OpenArm cube stacking.")
parser.add_argument(
    "--task",
    type=str,
    default="Nepher-OpenArm-CubeStack-v0",
    help="Gym environment ID.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=500,
    help="Number of simulation steps to run.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Remaining imports
# ---------------------------------------------------------------------------

import gymnasium as gym
import torch

import openarm_cube_stacking.tasks  # noqa: F401


def main() -> None:
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    print(f"[zero_agent] Task        : {args_cli.task}")
    print(f"[zero_agent] Num envs    : {args_cli.num_envs}")
    print(f"[zero_agent] Action space: {env.action_space}")

    obs, _ = env.reset()
    print("[zero_agent] Environment reset OK.")

    action_dim = env.action_space.shape[-1]
    zero_actions = torch.zeros(args_cli.num_envs, action_dim).to(env.unwrapped.device)

    for step in range(args_cli.num_steps):
        obs, rewards, terminated, truncated, info = env.step(zero_actions)

        if step % 100 == 0:
            mean_rew = rewards.mean().item()
            print(f"  step {step:5d}  |  mean reward: {mean_rew:+.4f}")

    env.close()
    print("[zero_agent] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
