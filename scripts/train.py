"""Legacy smoke-test script (kept for backwards compatibility).

For proper RSL-RL training, use scripts/rsl_rl/train.py instead.

This script runs a quick random-action smoke test to verify the environment
can be instantiated and stepped without crashing.
"""

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Register the new environments
import openarm_cube_stacking.tasks  # noqa: F401


def main():
    print("Smoke-testing Nepher-OpenArm-CubeStack-v0 …")
    env = gym.make("Nepher-OpenArm-CubeStack-v0", num_envs=4)

    obs, _ = env.reset()
    print(f"  obs shape  : {obs['policy'].shape}")
    print(f"  action dim : {env.action_space.shape}")

    for step in range(100):
        actions = torch.rand(4, env.action_space.shape[-1], device=env.unwrapped.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)

        if step % 25 == 0:
            print(f"  step {step:3d}  |  mean reward: {rewards.mean().item():.4f}")

    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
