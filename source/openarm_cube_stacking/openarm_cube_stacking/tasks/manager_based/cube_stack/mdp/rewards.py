"""Reward helpers for the OpenArm cube stacking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import (
    CUBE_SIZE,
    TABLE_TOP_Z,
    current_cube_position,
    current_cube_to_target,
    current_target_position,
    ee_to_current_cube,
    placed_cubes_mask,
    sequential_placed_count,
    stack_base_position,
    update_best_stacked_count,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_current_cube(env: ManagerBasedRLEnv, std: float = 0.08) -> torch.Tensor:
    """Reward the end-effector for approaching the current cube."""

    distance = torch.linalg.norm(ee_to_current_cube(env), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def lifting_current_cube(env: ManagerBasedRLEnv, minimal_height: float = TABLE_TOP_Z + 0.06) -> torch.Tensor:
    """Reward lifting the active cube away from the table surface."""

    cube_height = current_cube_position(env)[:, 2]
    return (cube_height > minimal_height).to(torch.float32)


def moving_current_cube_to_target(
    env: ManagerBasedRLEnv,
    std: float = 0.12,
    minimal_height: float = TABLE_TOP_Z + 0.05,
) -> torch.Tensor:
    """Reward moving the current cube toward its target stack position."""

    cube_pos = current_cube_position(env)
    target_delta = current_cube_to_target(env)
    distance = torch.linalg.norm(target_delta, dim=-1)
    lifted = cube_pos[:, 2] > minimal_height
    return lifted.to(torch.float32) * (1.0 - torch.tanh(distance / std))


def placing_current_cube(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.02,
    z_threshold: float = 0.015,
) -> torch.Tensor:
    """Reward accurate placement of the current cube at the target level."""

    cube_pos = current_cube_position(env)
    target_pos = current_target_position(env)
    xy_distance = torch.linalg.norm(cube_pos[:, :2] - target_pos[:, :2], dim=-1)
    z_distance = torch.abs(cube_pos[:, 2] - target_pos[:, 2])
    return ((xy_distance < xy_threshold) & (z_distance < z_threshold)).to(torch.float32)


def stack_success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse bonus when all five cubes are stacked in order."""

    return (sequential_placed_count(env) >= 5).to(torch.float32)


def cube_drop_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float = TABLE_TOP_Z - 0.10,
    workspace_radius: float = 0.40,
) -> torch.Tensor:
    """Penalty when cubes leave the tabletop workspace or fall below the table."""

    cube_positions = env.scene["cube_0"].data.root_pos_w.new_zeros((env.num_envs, 5, 3))
    for cube_index in range(5):
        cube_positions[:, cube_index] = env.scene[f"cube_{cube_index}"].data.root_pos_w[:, :3]

    base_xy = stack_base_position(env)[:, :2].unsqueeze(1)
    radial_distance = torch.linalg.norm(cube_positions[:, :, :2] - base_xy, dim=-1)
    below_table = cube_positions[:, :, 2] < minimum_height
    outside_workspace = radial_distance > workspace_radius
    return (below_table | outside_workspace).any(dim=1).to(torch.float32)


def stack_collapse_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty if the stable stack height regresses after making progress."""

    best_count = update_best_stacked_count(env)
    current_count = sequential_placed_count(env)
    return (best_count > current_count).to(torch.float32)


def upright_stack_bonus(env: ManagerBasedRLEnv, quat_w_threshold: float = 0.9) -> torch.Tensor:
    """Small bonus when cubes remain close to upright while stacked."""

    placed_mask = placed_cubes_mask(env)
    upright_mask = []
    for cube_index in range(5):
        upright_mask.append(env.scene[f"cube_{cube_index}"].data.root_quat_w[:, 0] > quat_w_threshold)
    upright = torch.stack(upright_mask, dim=1)
    return (placed_mask & upright).sum(dim=1).to(torch.float32) / 5.0
