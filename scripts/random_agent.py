"""Run random actions in the OpenArm cube stacking environment.

Usage::

    python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4

This script launches Isaac Sim, creates the environment, and steps it with
uniformly random actions for a fixed number of steps.  It is primarily used
to verify that the environment loads correctly and the physics simulation
is stable.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing (must happen before Isaac Sim is launched)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Random agent for OpenArm cube stacking.")
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
    default=1000,
    help="Number of simulation steps to run.",
)

# Add Isaac Sim / AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Remaining imports (after Isaac Sim is live)
# ---------------------------------------------------------------------------

import gymnasium as gym
import torch

import openarm_cube_stacking.tasks  # noqa: F401 – triggers gym.register()


def main() -> None:
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    print(f"[random_agent] Task        : {args_cli.task}")
    print(f"[random_agent] Num envs    : {args_cli.num_envs}")
    print(f"[random_agent] Action space: {env.action_space}")
    print(f"[random_agent] Obs space   : {env.observation_space}")

    obs, _ = env.reset()
    print("[random_agent] Environment reset OK.")

    for step in range(args_cli.num_steps):
        # Sample uniform-random actions in [-1, 1]
        actions = (
            torch.rand(args_cli.num_envs, env.action_space.shape[-1])
            * 2.0
            - 1.0
        ).to(env.unwrapped.device)

        obs, rewards, terminated, truncated, info = env.step(actions)

        if step % 100 == 0:
            mean_rew = rewards.mean().item()
            print(f"  step {step:5d}  |  mean reward: {mean_rew:+.4f}")

    env.close()
    print("[random_agent] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
