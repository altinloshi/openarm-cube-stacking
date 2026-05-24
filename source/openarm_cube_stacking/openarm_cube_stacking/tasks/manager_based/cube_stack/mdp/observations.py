"""Observation helper functions for the OpenArm cube stacking task."""

from __future__ import annotations

from typing import Sequence

import torch

CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{i}" for i in range(5))
NUM_CUBES = len(CUBE_NAMES)
CUBE_SIZE = 0.05
TABLE_TOP_Z = 0.0
STACK_BASE_DEFAULT = (0.55, 0.0, TABLE_TOP_Z + 0.5 * CUBE_SIZE)
STACK_XY_TOL = 0.03
STACK_Z_TOL = 0.03


def _ensure_stack_base(env) -> torch.Tensor:
    """Create a per-environment stack base tensor if one does not exist yet."""
    if not hasattr(env, "stack_target_base"):
        base = torch.tensor(STACK_BASE_DEFAULT, device=env.device, dtype=torch.float32)
        env.stack_target_base = base.repeat(env.num_envs, 1)
    return env.stack_target_base


def ee_position(env) -> torch.Tensor:
    """Returns end-effector position in each environment's local frame."""
    ee_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    return ee_pos_w - env.scene.env_origins


def ee_pose(env) -> torch.Tensor:
    """Returns end-effector pose as (x, y, z, qw, qx, qy, qz)."""
    ee_pos = ee_position(env)
    ee_quat = env.scene["ee_frame"].data.target_quat_w[:, 0, :]
    return torch.cat((ee_pos, ee_quat), dim=-1)


def cube_positions(env, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Returns stacked cube positions with shape (num_envs, num_cubes, 3)."""
    cube_pos_w = torch.stack([env.scene[name].data.root_pos_w for name in cube_names], dim=1)
    return cube_pos_w - env.scene.env_origins.unsqueeze(1)


def cube_orientations(env, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Returns stacked cube root orientations (quaternions)."""
    return torch.stack([env.scene[name].data.root_quat_w for name in cube_names], dim=1)


def cube_linear_velocities(env, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Returns stacked cube linear velocities with shape (num_envs, num_cubes, 3)."""
    return torch.stack([env.scene[name].data.root_lin_vel_w for name in cube_names], dim=1)


def stack_target_positions(env) -> torch.Tensor:
    """Computes all 5 target stack centers from the shared stack base."""
    base = _ensure_stack_base(env)
    z_offsets = torch.arange(NUM_CUBES, device=env.device, dtype=base.dtype) * CUBE_SIZE
    targets = base.unsqueeze(1).repeat(1, NUM_CUBES, 1)
    targets[:, :, 2] += z_offsets.unsqueeze(0)
    return targets


def cubes_in_target_mask(env, xy_tol: float = STACK_XY_TOL, z_tol: float = STACK_Z_TOL) -> torch.Tensor:
    """Boolean matrix indicating if each cube is close to its target slot."""
    cube_pos = cube_positions(env)
    targets = stack_target_positions(env)
    xy_ok = torch.linalg.norm(cube_pos[:, :, :2] - targets[:, :, :2], dim=-1) < xy_tol
    z_ok = torch.abs(cube_pos[:, :, 2] - targets[:, :, 2]) < z_tol
    return xy_ok & z_ok


def current_cube_index(env) -> torch.Tensor:
    """Returns index of first cube that is not yet placed, shape (num_envs, 1)."""
    placed = cubes_in_target_mask(env)
    missing = (~placed).to(torch.float32)
    first_missing = torch.argmax(missing, dim=1)
    all_placed = torch.all(placed, dim=1)
    last_index = torch.full_like(first_missing, NUM_CUBES - 1)
    current_index = torch.where(all_placed, last_index, first_missing)
    return current_index.unsqueeze(-1).to(torch.float32)


def current_cube_position(env) -> torch.Tensor:
    """Returns current cube position from sequential stacking logic."""
    cube_pos = cube_positions(env)
    index = current_cube_index(env).squeeze(-1).to(torch.long)
    batch = torch.arange(env.num_envs, device=env.device)
    return cube_pos[batch, index]


def current_target_position(env) -> torch.Tensor:
    """Returns target position for the current cube index."""
    targets = stack_target_positions(env)
    index = current_cube_index(env).squeeze(-1).to(torch.long)
    batch = torch.arange(env.num_envs, device=env.device)
    return targets[batch, index]


def ee_to_current_cube(env) -> torch.Tensor:
    """Returns vector from end-effector to current cube."""
    return current_cube_position(env) - ee_position(env)


def current_cube_to_target(env) -> torch.Tensor:
    """Returns vector from current cube to its target stack slot."""
    return current_target_position(env) - current_cube_position(env)


def all_cubes_placed_mask(env) -> torch.Tensor:
    """Returns whether each environment has all cubes in stack targets."""
    return torch.all(cubes_in_target_mask(env), dim=1)
