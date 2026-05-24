"""CLI argument helpers for the RSL-RL train/play scripts.

Mirrors the helper used in the upstream Isaac Lab RSL-RL workflow so that the
external task uses the same flag names (``--experiment_name``, ``--run_name``,
``--resume``, ``--load_run``, ``--checkpoint``, ``--logger`` ...).
"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL specific arguments to ``parser``."""
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    group.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name.")
    group.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    group.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
    group.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load.")
    group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Project name when using wandb or neptune.",
    )


def update_rsl_rl_cfg(agent_cfg: "RslRlBaseRunnerCfg", args_cli: argparse.Namespace) -> "RslRlBaseRunnerCfg":
    """Apply CLI overrides on top of the registered ``agent_cfg``."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10_000)
        agent_cfg.seed = args_cli.seed
    if getattr(args_cli, "resume", None) is not None:
        agent_cfg.resume = args_cli.resume
    if getattr(args_cli, "load_run", None) is not None:
        agent_cfg.load_run = args_cli.load_run
    if getattr(args_cli, "checkpoint", None) is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "experiment_name", None) is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if getattr(args_cli, "run_name", None) is not None:
        agent_cfg.run_name = args_cli.run_name
    if getattr(args_cli, "logger", None) is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> "RslRlBaseRunnerCfg":
    """Load the RSL-RL config registered for ``task_name`` and apply CLI overrides."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: "RslRlBaseRunnerCfg" = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(agent_cfg, args_cli)
