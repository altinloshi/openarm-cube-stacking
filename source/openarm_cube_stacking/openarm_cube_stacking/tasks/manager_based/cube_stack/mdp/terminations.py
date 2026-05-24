"""Termination helpers for the OpenArm cube stacking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import sequential_placed_count, stack_base_position, update_best_stacked_count

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_cubes_stacked(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate successfully when all five cubes are stacked in order."""

    return sequential_placed_count(env) >= 5


def cube_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.20,
    workspace_radius: float = 0.45,
) -> torch.Tensor:
    """Terminate if any cube falls far below the work surface or leaves the workspace."""

    cube_positions = env.scene["cube_0"].data.root_pos_w.new_zeros((env.num_envs, 5, 3))
    for cube_index in range(5):
        cube_positions[:, cube_index] = env.scene[f"cube_{cube_index}"].data.root_pos_w[:, :3]

    base_xy = stack_base_position(env)[:, :2].unsqueeze(1)
    radial_distance = torch.linalg.norm(cube_positions[:, :, :2] - base_xy, dim=-1)
    below_height = cube_positions[:, :, 2] < minimum_height
    outside_workspace = radial_distance > workspace_radius
    return (below_height | outside_workspace).any(dim=1)


def stack_collapsed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate if the already-achieved stack height regresses."""

    best_count = update_best_stacked_count(env)
    current_count = sequential_placed_count(env)
    return best_count > current_count


def invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate on NaN/Inf values in robot or cube state buffers."""

    robot_joint_pos = env.scene["robot"].data.joint_pos
    valid = torch.isfinite(robot_joint_pos).all(dim=1)
    for cube_index in range(5):
        cube_pos = env.scene[f"cube_{cube_index}"].data.root_pos_w[:, :3]
        valid = valid & torch.isfinite(cube_pos).all(dim=1)
    return ~valid
