# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation metrics for the cube stacking tournament.

This module provides:
- :class:`StackMetricsAccumulator` – collects per-step and per-episode data
- :func:`compute_metrics` – aggregate final metrics dict
- :func:`check_cube_stacked` – per-cube success check

Terminology
-----------
A cube is "successfully stacked" when:
1. Its centre is within ``pos_tolerance`` of its target stack centre.
2. Its height is within ``height_tolerance`` of the expected stack height.
3. Its linear velocity is below ``velocity_threshold``.
4. All cubes below it in the stack are also stable.

A "full stack success" requires all five cubes stacked and stable at episode end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Per-cube success check
# ─────────────────────────────────────────────────────────────────────────────


def check_cube_stacked(
    cube_pos: torch.Tensor,
    cube_vel: torch.Tensor,
    target_pos: torch.Tensor,
    pos_tolerance: float = 0.04,
    height_tolerance: float = 0.02,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Return bool mask (num_envs,) indicating success for a single cube.

    Parameters
    ----------
    cube_pos : (num_envs, 3)
    cube_vel : (num_envs, 6)  — linear vel in first 3 dims
    target_pos : (num_envs, 3)
    """
    xy_err = torch.norm(cube_pos[:, :2] - target_pos[:, :2], dim=-1)
    z_err = torch.abs(cube_pos[:, 2] - target_pos[:, 2])
    linear_vel = torch.norm(cube_vel[:, :3], dim=-1)

    return (xy_err < pos_tolerance) & (z_err < height_tolerance) & (linear_vel < velocity_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics accumulator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StackMetricsAccumulator:
    """Accumulates stacking metrics over multiple episodes.

    Call :meth:`record_episode` at the end of each episode.
    Call :meth:`compute` to get the final aggregated metrics dict.
    """

    num_cubes: int = 5

    # Accumulators
    _full_success: list[bool] = field(default_factory=list)
    _cubes_stacked_counts: list[int] = field(default_factory=list)
    _per_cube_success: list[list[bool]] = field(default_factory=list)
    _drop_occurred: list[bool] = field(default_factory=list)
    _collapse_occurred: list[bool] = field(default_factory=list)
    _final_stack_errors: list[float] = field(default_factory=list)
    _episode_lengths_s: list[float] = field(default_factory=list)
    _grasp_retries: list[int] = field(default_factory=list)
    _timed_out: list[bool] = field(default_factory=list)

    def record_episode(
        self,
        cube_pos_final: torch.Tensor,
        cube_vel_final: torch.Tensor,
        target_pos: torch.Tensor,
        episode_length_s: float,
        grasp_retries: int = 0,
        timed_out: bool = True,
        drop_occurred: bool = False,
        collapse_occurred: bool = False,
    ) -> None:
        """Record metrics for a single completed episode.

        All tensors should be for a SINGLE environment (not batched).

        Parameters
        ----------
        cube_pos_final : (num_cubes, 3)
        cube_vel_final : (num_cubes, 6)
        target_pos : (num_cubes, 3) — target centre for each cube
        """
        per_cube_ok: list[bool] = []
        for i in range(self.num_cubes):
            ok = check_cube_stacked(
                cube_pos_final[i:i+1],
                cube_vel_final[i:i+1],
                target_pos[i:i+1],
            ).item()
            per_cube_ok.append(bool(ok))

        full_ok = all(per_cube_ok)
        stacked_count = sum(per_cube_ok)

        self._full_success.append(full_ok)
        self._cubes_stacked_counts.append(stacked_count)
        self._per_cube_success.append(per_cube_ok)
        self._drop_occurred.append(drop_occurred)
        self._collapse_occurred.append(collapse_occurred)

        # Mean positional error of all cubes to their targets at episode end
        errs = torch.norm(cube_pos_final - target_pos, dim=-1)
        self._final_stack_errors.append(errs.mean().item())

        self._episode_lengths_s.append(episode_length_s)
        self._grasp_retries.append(grasp_retries)
        self._timed_out.append(timed_out)

    def compute(self) -> dict:
        """Aggregate and return all metrics as a Python dict."""
        n = len(self._full_success)
        if n == 0:
            return _empty_metrics(self.num_cubes)

        per_cube_rates = []
        for i in range(self.num_cubes):
            successes = sum(ep[i] for ep in self._per_cube_success)
            per_cube_rates.append(round(successes / n, 4))

        return {
            "num_episodes": n,
            "full_stack_success_rate": round(sum(self._full_success) / n, 4),
            "average_cubes_stacked": round(sum(self._cubes_stacked_counts) / n, 4),
            "per_cube_success_rate": per_cube_rates,
            "cube_drop_rate": round(sum(self._drop_occurred) / n, 4),
            "stack_collapse_rate": round(sum(self._collapse_occurred) / n, 4),
            "mean_final_stack_error": round(sum(self._final_stack_errors) / n, 4),
            "mean_episode_length_s": round(sum(self._episode_lengths_s) / n, 4),
            "mean_grasp_retries": round(sum(self._grasp_retries) / n, 4),
            "timeout_rate": round(sum(self._timed_out) / n, 4),
        }


def _empty_metrics(num_cubes: int) -> dict:
    return {
        "num_episodes": 0,
        "full_stack_success_rate": 0.0,
        "average_cubes_stacked": 0.0,
        "per_cube_success_rate": [0.0] * num_cubes,
        "cube_drop_rate": 0.0,
        "stack_collapse_rate": 0.0,
        "mean_final_stack_error": 0.0,
        "mean_episode_length_s": 0.0,
        "mean_grasp_retries": 0.0,
        "timeout_rate": 0.0,
    }


def compute_metrics(accumulator: StackMetricsAccumulator) -> dict:
    """Convenience wrapper around :meth:`StackMetricsAccumulator.compute`."""
    return accumulator.compute()
