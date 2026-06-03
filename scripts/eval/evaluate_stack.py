"""Deterministic tournament evaluation for OpenArm cube stacking.

Runs the deterministic eval environment (``Nepher-OpenArm-CubeStack-Eval-v0``),
executing the frozen LL policy + classical stack planner across the 30
pre-baked scenarios, and writes the tournament metrics to:

    logs/eval/openarm_cube_stack/<timestamp>/results.json

Example::

    python scripts/eval/evaluate_stack.py --task=Nepher-OpenArm-CubeStack-Eval-v0 \
        --num_envs=30 --episodes=30
"""

import argparse
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# Make the rsl_rl helper scripts importable (policy_paths, cli_args).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.join(_PROJECT_ROOT, "scripts", "rsl_rl"))

import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluate OpenArm cube stacking on deterministic scenarios.")
parser.add_argument("--num_envs", type=int, default=30, help="Number of parallel environments (one per scenario).")
parser.add_argument("--episodes", type=int, default=30, help="Total number of episodes to record.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-Eval-v0", help="Gym task ID.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=0, help="Environment and agent seed (reproducible).")
parser.add_argument("--out", type=str, default=None, help="Optional explicit results.json output path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Evaluation runs headless by default unless the user enabled a viewer.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import openarm_cube_stacking.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from openarm_cube_stacking.tasks.manager_based.cube_stack import tabletop  # noqa: E402
from openarm_cube_stacking.tasks.manager_based.cube_stack.eval import metrics as metrics_mod  # noqa: E402
from openarm_cube_stacking.tasks.manager_based.cube_stack.eval import scenarios as scenarios_mod  # noqa: E402

import policy_paths  # isort: skip  # noqa: E402


installed_version = metadata.version("rsl-rl-lib")


def _stack_targets(pose_term, num_cubes: int, device) -> torch.Tensor:
    """Per-env, per-cube stack-slot centres ``(N, M, 3)`` (world frame)."""
    stack_base = pose_term.stack_base_w
    z_off = torch.arange(num_cubes, device=device, dtype=torch.float32) * tabletop.CUBE_SIZE
    targets = stack_base[:, None, :].repeat(1, num_cubes, 1)
    targets[:, :, 2] += z_off[None, :]
    return targets


def _snapshot(unwrapped, cube_names, num_cubes: int):
    """Read the current stacking state from the unwrapped env."""
    device = unwrapped.device
    pos = torch.stack([unwrapped.scene[n].data.root_pos_w[:, :3] for n in cube_names], dim=1)
    vel = torch.stack([unwrapped.scene[n].data.root_lin_vel_w[:, :3] for n in cube_names], dim=1)
    pose_term = unwrapped.command_manager.get_term("ee_pose")
    targets = _stack_targets(pose_term, num_cubes, device)

    stacked = metrics_mod.cubes_stacked_mask(pos, vel, targets)
    stacked_prefix = metrics_mod.stacked_prefix_mask(stacked)
    stack_err = metrics_mod.final_stack_error(pos, targets)
    dropped = metrics_mod.cubes_dropped_mask(pos, min_height=tabletop.TABLE_TOP_Z - 0.08)
    retries = pose_term.planner._retry_count.clone()
    return stacked_prefix, stack_err, dropped, retries


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Run deterministic evaluation and write results.json."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else 0
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = policy_paths.sync_best_policy(
        agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        explicit_checkpoint=args_cli.checkpoint,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading LL policy checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    device = unwrapped.device
    cube_names = list(tabletop.CUBE_NAMES)
    num_cubes = len(cube_names)
    step_dt = unwrapped.step_dt

    evaluator = metrics_mod.StackEvaluator(num_cubes=num_cubes)

    # Per-env episode trackers.
    ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    max_stacked = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Safeguard on total control steps.
    episode_steps = int(unwrapped.max_episode_length)
    batches = (args_cli.episodes + num_envs - 1) // num_envs
    max_total_steps = episode_steps * (batches + 2)

    obs = env.get_observations()
    step_count = 0
    while simulation_app.is_running() and evaluator.num_episodes < args_cli.episodes:
        # Snapshot the (near-terminal) state BEFORE stepping/auto-reset.
        stacked_prefix, stack_err, dropped, retries = _snapshot(unwrapped, cube_names, num_cubes)
        cur_stacked = stacked_prefix.sum(dim=1)
        max_stacked = torch.maximum(max_stacked, cur_stacked)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        ep_steps += 1
        step_count += 1

        # Classify terminations from this step.
        tm = unwrapped.termination_manager
        timeout_fired = tm.get_term("time_out") if "time_out" in tm.active_terms else torch.zeros(num_envs, dtype=torch.bool, device=device)

        done_ids = torch.where(dones.bool())[0]
        for i in done_ids.tolist():
            if evaluator.num_episodes >= args_cli.episodes:
                break
            final_stacked = int(cur_stacked[i].item())
            collapsed = bool(final_stacked < int(max_stacked[i].item()) and int(max_stacked[i].item()) > 0)
            evaluator.record_episode(
                per_cube_stacked=stacked_prefix[i],
                dropped=bool(dropped[i].item()),
                collapsed=collapsed,
                stack_error=float(stack_err[i].item()),
                episode_len_s=float(ep_steps[i].item()) * step_dt,
                grasp_retries=float(retries[i].item()),
                timeout=bool(timeout_fired[i].item()),
            )
            # Reset trackers for the env that just restarted.
            ep_steps[i] = 0
            max_stacked[i] = 0

        if step_count >= max_total_steps:
            print(f"[WARN] Reached step safeguard ({max_total_steps}); stopping early.")
            break

    results = {
        "task": args_cli.task,
        "num_envs": int(num_envs),
        "episodes": int(evaluator.num_episodes),
        **evaluator.compute(),
        "scenarios": scenarios_mod.describe(),
    }

    # Write results.json.
    if args_cli.out is not None:
        out_path = os.path.abspath(args_cli.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(policy_paths.PROJECT_ROOT, "logs", "eval", "openarm_cube_stack", ts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Wrote evaluation results to: {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "scenarios"}, indent=2))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
