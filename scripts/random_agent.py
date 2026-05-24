"""Run the OpenArm cube stacking environment with random actions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "openarm_cube_stacking"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


parser = argparse.ArgumentParser(description="Run random actions in the OpenArm cube stacking task.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of vectorized environments.")
parser.add_argument("--steps", type=int, default=500, help="Number of control steps to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm_cube_stacking.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    single_action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device, dtype=torch.float32)
    single_action_shape = single_action.shape if single_action.ndim > 0 else (1,)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            actions = torch.rand((env.unwrapped.num_envs, *single_action_shape), device=env.unwrapped.device) * 2.0 - 1.0
            _, rewards, terminated, truncated, _ = env.step(actions)
            if step % 100 == 0:
                done_fraction = (terminated | truncated).float().mean().item()
                print(
                    f"step={step:04d} mean_reward={rewards.mean().item(): .4f} "
                    f"done_fraction={done_fraction: .3f}"
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
