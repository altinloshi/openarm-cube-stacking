"""Play / evaluate a trained RSL-RL policy on the OpenArm cube-stack play env.

Usage::

    python scripts/rsl_rl/play.py \
        --task=Nepher-OpenArm-CubeStack-Play-v0 \
        --checkpoint=/path/to/model.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play a trained RSL-RL agent on the OpenArm cube-stack env.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-Play-v0", help="Gym env id.")
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of parallel envs.")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of env steps to play.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import openarm_cube_stacking.tasks  # noqa: E402, F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint is not None and os.path.isfile(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if simulation_app.is_running() is False:
                break
            actions = policy(obs)
            obs, *_ = env.step(actions)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
