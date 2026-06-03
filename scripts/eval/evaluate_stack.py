"""Evaluate deterministic OpenArm cube-stacking scenarios."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate OpenArm cube stacking tournament scenarios.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-Eval-v0")
parser.add_argument("--num_envs", type=int, default=30)
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--checkpoint", type=str, default=None, help="Optional LL checkpoint to sync into best_policy/.")
parser.add_argument("--allow_missing_policy", action="store_true", help="Run zero actions if no LL checkpoint is available.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import openarm_cube_stacking.tasks  # noqa: F401, E402

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.append(str(SCRIPT_DIR))
import policy_paths  # noqa: E402

from openarm_cube_stacking.tasks.manager_based.cube_stack.eval.metrics import summarize_episode_batch  # noqa: E402


def _load_policy(env, agent_cfg, checkpoint: str | None, allow_missing: bool):
    try:
        ckpt = policy_paths.sync_best_policy(
            agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint, explicit_checkpoint=checkpoint
        )
    except FileNotFoundError:
        if not allow_missing:
            raise
        print("[WARN] No LL policy checkpoint found; evaluating with zero actions for smoke testing.")
        return None
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    return runner.get_inference_policy(device=env.unwrapped.device)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.seed = 2026
    env = gym.make(args_cli.task, cfg=env_cfg)

    # The env registration stores the LL runner config. Load it through gym metadata
    # by constructing an RSL wrapper with a small default config object.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.experiment_name = getattr(agent_cfg, "experiment_name", "openarm_cube_stack_ll")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = _load_policy(wrapped, agent_cfg, args_cli.checkpoint, args_cli.allow_missing_policy)

    obs = wrapped.get_observations()
    completed = 0
    timeout_accum = torch.zeros(wrapped.unwrapped.num_envs, dtype=torch.bool, device=wrapped.unwrapped.device)
    max_steps = int(env_cfg.episode_length_s / wrapped.unwrapped.step_dt) * max(1, args_cli.episodes // args_cli.num_envs + 1)
    for _ in range(max_steps):
        with torch.inference_mode():
            if policy is None:
                actions = torch.zeros(wrapped.action_space.shape, device=wrapped.unwrapped.device)
            else:
                actions = policy(obs)
            obs, _rew, dones, _info = wrapped.step(actions)
            timeout_accum |= wrapped.unwrapped.reset_time_outs
            completed += int(dones.sum().item())
            if completed >= args_cli.episodes:
                break

    metrics = summarize_episode_batch(wrapped.unwrapped, args_cli.task, args_cli.episodes, timeout_mask=timeout_accum).to_dict()
    out_dir = Path("logs/eval/openarm_cube_stack") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[INFO] Wrote evaluation results: {out_path}")
    print(json.dumps(metrics, indent=2))
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
