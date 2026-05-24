"""Train an RL agent on the OpenArm cube-stack task with RSL-RL.

Usage::

    python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless

This script follows the same overall structure as the upstream Isaac Lab
``scripts/reinforcement_learning/rsl_rl/train.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Local sibling import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL on the OpenArm cube-stack env.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-v0", help="Gym env id.")
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of parallel envs.")
parser.add_argument("--seed", type=int, default=None, help="Environment / agent seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override RL training iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recorded videos (steps).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Launch sim BEFORE importing isaaclab modules.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------ rest after sim launch ------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import openarm_cube_stacking.tasks  # noqa: E402, F401  (registers envs)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main() -> None:
    # Resolve env/agent configs from the registry.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed

    # Logging dirs.
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create environment.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
