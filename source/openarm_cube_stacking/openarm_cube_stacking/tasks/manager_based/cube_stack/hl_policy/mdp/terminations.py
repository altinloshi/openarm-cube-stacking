from __future__ import annotations

"""Termination terms for the HL cube-stacking environment.

* :func:`all_cubes_stacked`  — success: every cube placed at its stack slot,
  upright-stable, and not moving.
* :func:`stack_failed`       — failure: a cube fell off the table, or the
  planner aborted after exhausting grasp retries (eval-strict mode).
* ``time_out`` is provided by the standard MDP library.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject

from ...tabletop import CUBE_NAMES, CUBE_SIZE, TABLE_TOP_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cube_states(
    env: ManagerBasedRLEnv, cube_names: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cube positions ``(N, M, 3)`` and linear velocities ``(N, M, 3)``."""
    pos = []
    vel = []
    for name in cube_names:
        cube: RigidObject = env.scene[name]
        pos.append(cube.data.root_pos_w[:, :3])
        vel.append(cube.data.root_lin_vel_w[:, :3])
    return torch.stack(pos, dim=1), torch.stack(vel, dim=1)


def stack_target_positions(env: ManagerBasedRLEnv, pose_cmd_name: str, num_cubes: int) -> torch.Tensor:
    """Per-env, per-cube stack-slot centres ``(N, M, 3)`` in world frame."""
    term = env.command_manager.get_term(pose_cmd_name)
    stack_base = term.stack_base_w  # (N, 3)
    z_offsets = torch.arange(num_cubes, device=env.device, dtype=torch.float32) * CUBE_SIZE
    targets = stack_base[:, None, :].repeat(1, num_cubes, 1)
    targets[:, :, 2] += z_offsets[None, :]
    return targets


def cube_stacked_mask(
    cube_pos: torch.Tensor,
    cube_vel: torch.Tensor,
    target_pos: torch.Tensor,
    xy_tol: float = 0.03,
    z_tol: float = 0.025,
    vel_tol: float = 0.05,
) -> torch.Tensor:
    """Per-cube boolean mask: placed within tolerance, at the right height, still.

    Args:
        cube_pos:   ``(N, M, 3)`` cube centres (world).
        cube_vel:   ``(N, M, 3)`` cube linear velocities (world).
        target_pos: ``(N, M, 3)`` stack-slot centres (world).
        xy_tol:     allowed horizontal offset from the slot centre (m).
        z_tol:      allowed height error from the expected stack height (m).
        vel_tol:    maximum speed for a cube to count as "stable" (m/s).
    """
    xy_err = torch.norm(cube_pos[..., :2] - target_pos[..., :2], dim=-1)
    z_err = (cube_pos[..., 2] - target_pos[..., 2]).abs()
    speed = torch.norm(cube_vel, dim=-1)
    return (xy_err < xy_tol) & (z_err < z_tol) & (speed < vel_tol)


def all_cubes_stacked(
    env: ManagerBasedRLEnv,
    cube_names: Sequence[str] = CUBE_NAMES,
    pose_cmd_name: str = "ee_pose",
    xy_tol: float = 0.03,
    z_tol: float = 0.025,
    vel_tol: float = 0.05,
) -> torch.Tensor:
    """Success: all cubes stacked, stable, and the planner has finished."""
    cube_pos, cube_vel = _cube_states(env, cube_names)
    target_pos = stack_target_positions(env, pose_cmd_name, len(cube_names))
    placed = cube_stacked_mask(cube_pos, cube_vel, target_pos, xy_tol, z_tol, vel_tol)
    term = env.command_manager.get_term(pose_cmd_name)
    planner_done = term.planner.is_fully_done()
    return placed.all(dim=1) & planner_done


def stack_failed(
    env: ManagerBasedRLEnv,
    cube_names: Sequence[str] = CUBE_NAMES,
    pose_cmd_name: str = "ee_pose",
    min_height: float = TABLE_TOP_Z - 0.08,
) -> torch.Tensor:
    """Failure: a cube fell below the table, or the planner aborted."""
    cube_pos, _ = _cube_states(env, cube_names)
    below_table = (cube_pos[:, :, 2] < min_height).any(dim=1)
    term = env.command_manager.get_term(pose_cmd_name)
    planner_failed = term.planner._episode_failed
    return below_table | planner_failed
