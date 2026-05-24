"""CLI argument helpers for RSL-RL train and play scripts."""

from __future__ import annotations

import argparse


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL specific arguments to an argument parser.

    Args:
        parser: The argument parser to extend.
    """
    group = parser.add_argument_group("RSL-RL", "Arguments for the RSL-RL runner.")

    group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override the experiment name set in the PPO runner config.",
    )
    group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix appended to the experiment directory timestamp.",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in --load_run.",
    )
    group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Subdirectory name (or 'latest') under the experiment log folder "
             "from which to load a checkpoint.",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to a checkpoint .pt file to load.",
    )
    group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "neptune"],
        help="Logging backend (defaults to tensorboard if not set).",
    )
    group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Project name for W&B / Neptune logging.",
    )
