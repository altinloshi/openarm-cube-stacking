from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import (
    CUBE_SIZE,
    TABLE_TOP_Z,
    _cube_orientations_w,
    _cube_positions_w,
    _current_cube_indices,
    _gather_current,
    _placed_mask,
    _target_positions_w,
    end_effector_position,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_current_cube(env: ManagerBasedRLEnv, std: float = 0.08) -> torch.Tensor:
    """Reward proximity between the end-effector and the current cube."""
    cube_pos = _gather_current(_cube_positions_w(env), _current_cube_indices(env))
    distance = torch.norm(cube_pos - end_effector_position(env), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def lifting_current_cube(env: ManagerBasedRLEnv, minimal_height: float = 0.04) -> torch.Tensor:
    """Reward lifting the current cube above the tabletop."""
    cube_pos = _gather_current(_cube_positions_w(env), _current_cube_indices(env))
    lifted_height = cube_pos[:, 2] - (TABLE_TOP_Z + CUBE_SIZE / 2.0)
    return (lifted_height > minimal_height).to(torch.float32)


def moving_current_cube_to_target(env: ManagerBasedRLEnv, std: float = 0.20, minimal_height: float = 0.02) -> torch.Tensor:
    """Reward moving the current cube toward its sequential stack target."""
    current_idx = _current_cube_indices(env)
    cube_pos = _gather_current(_cube_positions_w(env), current_idx)
    target_pos = _gather_current(_target_positions_w(env), current_idx)
    distance = torch.norm(cube_pos - target_pos, dim=-1)
    lifted = cube_pos[:, 2] > (TABLE_TOP_Z + CUBE_SIZE / 2.0 + minimal_height)
    # Cube 0's target is already on the table; do not require a lift gate for that first placement.
    first_cube = current_idx == 0
    return (lifted | first_cube).to(torch.float32) * (1.0 - torch.tanh(distance / std))


def placing_current_cube(env: ManagerBasedRLEnv, threshold: float = 0.03) -> torch.Tensor:
    """Reward placing the current cube at its target stack location."""
    current_idx = _current_cube_indices(env)
    cube_pos = _gather_current(_cube_positions_w(env), current_idx)
    target_pos = _gather_current(_target_positions_w(env), current_idx)
    distance = torch.norm(cube_pos - target_pos, dim=-1)
    return (distance < threshold).to(torch.float32)


def stack_success_bonus(env: ManagerBasedRLEnv, threshold: float = 0.035) -> torch.Tensor:
    """Reward successful completion of all five placements."""
    return _placed_mask(env, threshold=threshold).all(dim=1).to(torch.float32)


def cube_drop_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = TABLE_TOP_Z - 0.08,
    workspace_radius: float = 0.90,
) -> torch.Tensor:
    """Return one when any cube falls below the table or outside the local workspace."""
    cube_pos = _cube_positions_w(env)
    env_origins = env.scene.env_origins.to(env.device)[:, None, :]
    local_xy = cube_pos[:, :, :2] - env_origins[:, :, :2]
    below_table = cube_pos[:, :, 2] < min_height
    outside_workspace = torch.norm(local_xy, dim=-1) > workspace_radius
    return (below_table | outside_workspace).any(dim=1).to(torch.float32)


def stack_collapse_penalty(env: ManagerBasedRLEnv, threshold: float = 0.055, min_upright_z: float = 0.94) -> torch.Tensor:
    """Return one when previously placed cubes drift or tilt significantly."""
    cube_pos = _cube_positions_w(env)
    cube_quat = _cube_orientations_w(env)
    target_pos = _target_positions_w(env)
    current_idx = _current_cube_indices(env)

    cube_ids = torch.arange(cube_pos.shape[1], device=env.device).unsqueeze(0)
    previous_cube_mask = cube_ids < current_idx.unsqueeze(1)
    distance = torch.norm(cube_pos - target_pos, dim=-1)

    qw, qx, qy, qz = cube_quat.unbind(dim=-1)
    z_axis_z = 1.0 - 2.0 * (qx.square() + qy.square())
    tilted = z_axis_z < min_upright_z

    collapsed = previous_cube_mask & ((distance > threshold) | tilted)
    return collapsed.any(dim=1).to(torch.float32)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large normalized actions."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize robot joint velocities."""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

