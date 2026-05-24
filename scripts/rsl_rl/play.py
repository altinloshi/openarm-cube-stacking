"""Run a trained RSL-RL policy in the OpenArm cube stacking play environment.

Usage::

    python scripts/rsl_rl/play.py \\
        --task=Nepher-OpenArm-CubeStack-Play-v0 \\
        --checkpoint=logs/rsl_rl/openarm_cube_stack/<run>/model_5000.pt

    # Record a video
    python scripts/rsl_rl/play.py \\
        --task=Nepher-OpenArm-CubeStack-Play-v0 \\
        --checkpoint=/path/to/model.pt \\
        --video \\
        --video_length=500

The play environment uses fewer environments and disables observation noise.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from cli_args import add_rsl_rl_args  # noqa: E402

parser = argparse.ArgumentParser(description="Play OpenArm cube stacking with trained RSL-RL policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Nepher-OpenArm-CubeStack-Play-v0",
    help="Gym environment ID (should be the play variant).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override number of environments.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000,
    help="Number of simulation steps to run.",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record a video of the episode.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Number of steps to record.",
)

add_rsl_rl_args(parser)
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
import importlib

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import openarm_cube_stacking.tasks  # noqa: F401


def _load_class_from_entry_point(entry_point: str):
    module_path, class_name = entry_point.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    spec = gym.spec(args_cli.task)
    kw = spec.kwargs

    EnvCfgClass = _load_class_from_entry_point(kw["env_cfg_entry_point"])
    RunnerCfgClass = _load_class_from_entry_point(kw["rsl_rl_cfg_entry_point"])

    env_cfg = EnvCfgClass()
    runner_cfg: RslRlOnPolicyRunnerCfg = RunnerCfgClass()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        from gymnasium.wrappers import RecordVideo  # type: ignore[import]

        video_dir = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name, "play_videos")
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            video_length=args_cli.video_length,
        )

    env = RslRlVecEnvWrapper(env)

    # Build runner and load checkpoint
    runner = OnPolicyRunner(
        env, runner_cfg.to_dict(), log_dir=None, device=args_cli.device
    )

    checkpoint_path = args_cli.checkpoint
    if checkpoint_path is None:
        raise ValueError(
            "Please specify --checkpoint=/path/to/model.pt  "
            "(or use --load_run to pick the latest from a run directory)."
        )

    print(f"[play] Loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    # --- Inference loop ---
    obs, _ = env.get_observations()
    total_steps = args_cli.num_steps
    for step in range(total_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs, rewards, dones, info = env.step(actions)

        if step % 100 == 0:
            print(f"  step {step:5d}  |  mean reward: {rewards.mean().item():+.4f}")

    env.close()
    print("[play] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
