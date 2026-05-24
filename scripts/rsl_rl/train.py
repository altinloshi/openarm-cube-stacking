"""Train the OpenArm 5-cube stacking policy with RSL-RL (PPO).

Usage::

    # Headless training (recommended for servers)
    python scripts/rsl_rl/train.py \\
        --task=Nepher-OpenArm-CubeStack-v0 \\
        --headless

    # Interactive training (with Isaac Sim viewport)
    python scripts/rsl_rl/train.py \\
        --task=Nepher-OpenArm-CubeStack-v0 \\
        --num_envs=512

    # Resume from latest checkpoint
    python scripts/rsl_rl/train.py \\
        --task=Nepher-OpenArm-CubeStack-v0 \\
        --headless \\
        --resume \\
        --load_run=<run_folder_name>

    # Override max iterations
    python scripts/rsl_rl/train.py \\
        --task=Nepher-OpenArm-CubeStack-v0 \\
        --headless \\
        --max_iterations=2000
"""

from __future__ import annotations

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# CLI parsing must happen before AppLauncher (which parses --headless etc.)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

# Import from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from cli_args import add_rsl_rl_args  # noqa: E402

parser = argparse.ArgumentParser(description="Train OpenArm cube stacking with RSL-RL PPO.")
parser.add_argument(
    "--task",
    type=str,
    default="Nepher-OpenArm-CubeStack-v0",
    help="Gym environment ID to train.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override the number of parallel environments.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Override the maximum number of training iterations.",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video during training.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Number of steps per video clip.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Record a video every N training iterations.",
)

add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless when video recording is disabled (optional convenience)
if not args_cli.video:
    args_cli.headless = getattr(args_cli, "headless", False)

# ---------------------------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All remaining imports go AFTER the app is started
# ---------------------------------------------------------------------------
import datetime
import importlib

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Register our custom environments
import openarm_cube_stacking.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_class_from_entry_point(entry_point: str):
    """Load a Python class from an ``module:ClassName`` entry-point string."""
    module_path, class_name = entry_point.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_log_dir(experiment_name: str, run_name: str | None = None) -> str:
    """Build a timestamped log directory path."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{run_name}" if run_name else ""
    return os.path.join("logs", "rsl_rl", experiment_name, f"{timestamp}{suffix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Retrieve env config and RSL-RL config from gym registry ---
    spec = gym.spec(args_cli.task)
    kw = spec.kwargs

    env_cfg_entry = kw["env_cfg_entry_point"]
    rsl_cfg_entry = kw["rsl_rl_cfg_entry_point"]

    EnvCfgClass = _load_class_from_entry_point(env_cfg_entry)
    RunnerCfgClass = _load_class_from_entry_point(rsl_cfg_entry)

    env_cfg: ManagerBasedRLEnv = EnvCfgClass()
    runner_cfg: RslRlOnPolicyRunnerCfg = RunnerCfgClass()

    # Apply CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        runner_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        runner_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        runner_cfg.experiment_name = args_cli.experiment_name

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        from gymnasium.wrappers import RecordVideo  # type: ignore[import]

        log_dir = _get_log_dir(runner_cfg.experiment_name, args_cli.run_name)
        env = RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
        )

    # --- Wrap for RSL-RL ---
    env = RslRlVecEnvWrapper(env)

    # --- Set up logging directory ---
    log_dir = _get_log_dir(runner_cfg.experiment_name, args_cli.run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] Logging to: {log_dir}")

    # --- Build runner ---
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)

    # Resume or load checkpoint
    if args_cli.resume or args_cli.load_run is not None:
        resume_path = args_cli.checkpoint
        if resume_path is None and args_cli.load_run is not None:
            run_dir = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name, args_cli.load_run)
            # find latest model in that run
            ckpts = sorted(
                [f for f in os.listdir(run_dir) if f.endswith(".pt")],
                key=lambda f: int(f.split("_")[-1].replace(".pt", "")) if f[0].isdigit() else 0,
            )
            resume_path = os.path.join(run_dir, ckpts[-1]) if ckpts else None
        if resume_path:
            print(f"[train] Loading checkpoint: {resume_path}")
            runner.load(resume_path)

    # --- Train ---
    print_dict(runner_cfg.to_dict())
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
