"""Termination helper functions for OpenArm cube stacking."""

from __future__ import annotations

import torch

from . import observations as obs


def all_cubes_stacked(env) -> torch.Tensor:
    """Episode success condition."""
    return obs.all_cubes_placed_mask(env)


def cube_dropped(
    env,
    minimum_height: float = obs.TABLE_TOP_Z - 0.08,
    workspace_radius: float = 0.45,
) -> torch.Tensor:
    """Terminate when cubes leave workspace bounds or drop below table."""
    cube_pos = obs.cube_positions(env)
    below_min_height = cube_pos[:, :, 2] < minimum_height
    stack_base = obs._ensure_stack_base(env)  # pylint: disable=protected-access
    radial_offset = torch.linalg.norm(cube_pos[:, :, :2] - stack_base[:, None, :2], dim=-1)
    outside_workspace = radial_offset > workspace_radius
    return (below_min_height | outside_workspace).any(dim=1)


def stack_collapsed(env, xy_tol: float = 0.05, z_tol: float = 0.06) -> torch.Tensor:
    """Terminate when previously completed cube levels become unstable."""
    placed = obs.cubes_in_target_mask(env, xy_tol=xy_tol, z_tol=z_tol)
    current_index = obs.current_cube_index(env).squeeze(-1).to(torch.long)
    cube_idx = torch.arange(obs.NUM_CUBES, device=env.device).unsqueeze(0)
    should_be_placed = cube_idx < current_index.unsqueeze(-1)
    return (should_be_placed & (~placed)).any(dim=1)


def robot_or_cube_invalid_state(env) -> torch.Tensor:
    """Terminate if simulation returns NaN/Inf for robot or cube state."""
    robot_joint_pos = env.scene["robot"].data.joint_pos
    cube_pos = obs.cube_positions(env)
    robot_invalid = ~torch.isfinite(robot_joint_pos).all(dim=1)
    cube_invalid = ~torch.isfinite(cube_pos).all(dim=(1, 2))
    return robot_invalid | cube_invalid
