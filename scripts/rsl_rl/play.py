"""Play the OpenArm cube stacking task with a trained RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "source" / "openarm_cube_stacking"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description="Play a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record one video during play.")
parser.add_argument("--video_length", type=int, default=500, help="Recorded video length in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-Play-v0", help="Gym task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Gym registry key for the RL agent configuration.",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
parser.add_argument("--real_time", action="store_true", default=False, help="Sleep to match environment step time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

installed_rsl_rl_version = metadata.version("rsl-rl-lib")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import openarm_cube_stacking.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Load a trained policy and run inference in the play environment."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env_cfg.seed = agent_cfg.seed
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(env_cfg.log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    timestep = 0
    step_dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.real_time:
            sleep_time = step_dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
