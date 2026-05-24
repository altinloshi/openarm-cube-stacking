"""CLI helpers for RSL-RL train/play scripts."""

from __future__ import annotations

import argparse
import random

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Append common RSL-RL arguments."""
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL.")
    group.add_argument("--experiment_name", type=str, default=None)
    group.add_argument("--run_name", type=str, default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", type=str, default=None)
    group.add_argument("--checkpoint", type=str, default=None)
    group.add_argument("--seed", type=int, default=None)


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace):
    """Load and update RSL-RL runner config from task registry."""
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(agent_cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Override runner config fields from CLI arguments."""
    if getattr(args_cli, "seed", None) is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if getattr(args_cli, "resume", None) is not None:
        agent_cfg.resume = args_cli.resume
    if getattr(args_cli, "load_run", None):
        agent_cfg.load_run = args_cli.load_run
    if getattr(args_cli, "checkpoint", None):
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "experiment_name", None):
        agent_cfg.experiment_name = args_cli.experiment_name
    if getattr(args_cli, "run_name", None):
        agent_cfg.run_name = args_cli.run_name
    return agent_cfg
