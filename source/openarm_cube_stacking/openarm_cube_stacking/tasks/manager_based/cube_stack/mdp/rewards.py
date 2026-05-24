"""Reward helper functions for OpenArm 5-cube stacking."""

from __future__ import annotations

import torch

from . import observations as obs


def reaching_current_cube(env, std: float = 0.1) -> torch.Tensor:
    """Dense reward encouraging end-effector approach toward current cube."""
    dist = torch.linalg.norm(obs.ee_to_current_cube(env), dim=-1)
    return torch.exp(-dist / std)


def lifting_current_cube(env, minimal_height: float = obs.TABLE_TOP_Z + obs.CUBE_SIZE * 1.3) -> torch.Tensor:
    """Binary reward when the current cube is lifted from the table."""
    current_cube_height = obs.current_cube_position(env)[:, 2]
    return (current_cube_height > minimal_height).to(torch.float32)


def moving_current_cube_to_target(env, std: float = 0.15) -> torch.Tensor:
    """Dense reward for carrying the active cube toward its stack slot."""
    dist = torch.linalg.norm(obs.current_cube_to_target(env), dim=-1)
    return torch.exp(-dist / std)


def placing_current_cube(
    env,
    position_tolerance: float = 0.025,
    velocity_tolerance: float = 0.25,
) -> torch.Tensor:
    """Reward when the current cube is near target and mostly settled."""
    cube_to_target_dist = torch.linalg.norm(obs.current_cube_to_target(env), dim=-1)
    current_index = obs.current_cube_index(env).squeeze(-1).to(torch.long)
    cube_lin_vel = obs.cube_linear_velocities(env)
    batch = torch.arange(env.num_envs, device=env.device)
    current_cube_vel = torch.linalg.norm(cube_lin_vel[batch, current_index], dim=-1)
    return ((cube_to_target_dist < position_tolerance) & (current_cube_vel < velocity_tolerance)).to(torch.float32)


def stack_success_bonus(env) -> torch.Tensor:
    """Terminal success bonus once all five cubes are stacked."""
    return obs.all_cubes_placed_mask(env).to(torch.float32)


def action_penalty(env) -> torch.Tensor:
    """L2 action penalty to smooth policy outputs."""
    actions = env.action_manager.action
    return torch.sum(actions.square(), dim=-1)


def joint_velocity_penalty(env) -> torch.Tensor:
    """L2 penalty on robot joint velocities."""
    joint_vel = env.scene["robot"].data.joint_vel
    return torch.sum(joint_vel.square(), dim=-1)


def cube_drop_penalty(env, min_height: float = obs.TABLE_TOP_Z - 0.08) -> torch.Tensor:
    """Penalty when any cube falls substantially below table top."""
    dropped = obs.cube_positions(env)[:, :, 2] < min_height
    return dropped.any(dim=1).to(torch.float32)


def stack_collapse_penalty(env, xy_tol: float = 0.05, z_tol: float = 0.06) -> torch.Tensor:
    """Penalty when already-placed cubes move away from their stack slots."""
    placed = obs.cubes_in_target_mask(env, xy_tol=xy_tol, z_tol=z_tol)
    current_index = obs.current_cube_index(env).squeeze(-1).to(torch.long)
    cube_idx = torch.arange(obs.NUM_CUBES, device=env.device).unsqueeze(0)
    should_be_placed = cube_idx < current_index.unsqueeze(-1)
    collapsed = should_be_placed & (~placed)
    return collapsed.any(dim=1).to(torch.float32)
