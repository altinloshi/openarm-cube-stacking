"""Deterministic evaluation script for the cube stacking tournament.

Runs the HL classical planner + LL policy across 30 deterministic scenarios
and writes a JSON results file.

Usage
-----
    python scripts/eval/evaluate_stack.py \\
        --task=Nepher-OpenArm-CubeStack-Eval-v0 \\
        --num_envs=30 \\
        --episodes=30

Output
------
    logs/eval/openarm_cube_stack/<timestamp>/results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate OpenArm cube stacking.")
parser.add_argument(
    "--task",
    type=str,
    default="Nepher-OpenArm-CubeStack-Eval-v0",
    help="Gym task ID for evaluation.",
)
parser.add_argument("--num_envs", type=int, default=30, help="Number of parallel environments.")
parser.add_argument("--episodes", type=int, default=30, help="Total episodes to evaluate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to LL policy checkpoint.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Override output directory for results.json.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import openarm_cube_stacking.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from openarm_cube_stacking.tasks.manager_based.cube_stack.eval.metrics import (  # noqa: E402
    StackMetricsAccumulator,
    check_cube_stacked,
)
from openarm_cube_stacking.tasks.manager_based.cube_stack.openarm_lift_style_scene_cfg import (  # noqa: E402
    CUBE_NAMES,
    CUBE_SIZE,
)


installed_version = metadata.version("rsl-rl-lib")


def _stack_target_positions(stack_base_pos_w: torch.Tensor, num_cubes: int = 5) -> torch.Tensor:
    """Compute target positions for all cubes given stack base. (num_envs, num_cubes, 3)"""
    z_offsets = torch.arange(num_cubes, dtype=torch.float32, device=stack_base_pos_w.device) * CUBE_SIZE
    targets = stack_base_pos_w[:, None, :].expand(-1, num_cubes, -1).clone()
    targets[:, :, 2] += z_offsets[None, :]
    return targets


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = 0  # deterministic

    # Load LL policy checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # Try best_policy first, then fall back to latest log
        best = os.path.join("best_policy", "best_policy.pt")
        if os.path.isfile(best):
            resume_path = os.path.abspath(best)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Using policy checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── Output path ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.output_dir:
        out_dir = os.path.abspath(args_cli.output_dir)
    else:
        out_dir = os.path.join("logs", "eval", "openarm_cube_stack", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    accumulator = StackMetricsAccumulator(num_cubes=len(CUBE_NAMES))
    episodes_done = 0
    max_episodes = args_cli.episodes

    obs = env.get_observations()
    env.reset()

    step_dt = env.unwrapped.step_dt
    planner_retries = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)
    ep_length = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)

    print(f"[INFO]: Starting evaluation – {max_episodes} episodes across {args_cli.num_envs} envs")

    while episodes_done < max_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

        ep_length += step_dt

        # Collect metrics for done environments
        for env_idx in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
            if episodes_done >= max_episodes:
                break

            unwrapped = env.unwrapped
            # Cube final positions and velocities
            cube_pos = torch.stack(
                [unwrapped.scene[name].data.root_pos_w[env_idx] for name in CUBE_NAMES]
            )  # (num_cubes, 3)
            cube_vel = torch.stack(
                [unwrapped.scene[name].data.root_lin_vel_w[env_idx] for name in CUBE_NAMES]
            )  # (num_cubes, 3)
            # Pad velocity to 6 dims for the metric function
            cube_vel_6 = torch.cat([cube_vel, torch.zeros_like(cube_vel)], dim=-1)

            stack_base = getattr(unwrapped, "stack_base_pos_w", None)
            if stack_base is not None:
                sb = stack_base[env_idx:env_idx+1].expand(len(CUBE_NAMES), -1)
                z_offsets = torch.arange(len(CUBE_NAMES), device=sb.device, dtype=torch.float32) * CUBE_SIZE
                targets = sb.clone()
                targets[:, 2] += z_offsets
            else:
                targets = torch.zeros(len(CUBE_NAMES), 3, device=unwrapped.device)

            # Get retry count from planner if available
            retries_val = int(planner_retries[env_idx].item())

            accumulator.record_episode(
                cube_pos_final=cube_pos,
                cube_vel_final=cube_vel_6,
                target_pos=targets,
                episode_length_s=float(ep_length[env_idx].item()),
                grasp_retries=retries_val,
                timed_out=bool(dones[env_idx].item()),
            )
            episodes_done += 1
            ep_length[env_idx] = 0.0
            planner_retries[env_idx] = 0

            if episodes_done % 5 == 0:
                print(f"[INFO]: Episodes completed: {episodes_done}/{max_episodes}")

    env.close()

    # ── Compute and write results ─────────────────────────────────────────────
    metrics = accumulator.compute()
    metrics["task"] = args_cli.task
    metrics["num_envs"] = args_cli.num_envs
    metrics["episodes"] = args_cli.episodes

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO]: Evaluation complete. Results written to: {results_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
    simulation_app.close()
