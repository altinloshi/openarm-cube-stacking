from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import TABLE_TOP_Z, _cube_orientations_w, _cube_positions_w, _current_cube_indices, _placed_mask, _target_positions_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_cubes_stacked(env: ManagerBasedRLEnv, threshold: float = 0.035) -> torch.Tensor:
    """Terminate successfully when all five cubes are close to their target stack positions."""
    return _placed_mask(env, threshold=threshold).all(dim=1)


def cube_dropped(
    env: ManagerBasedRLEnv,
    min_height: float = TABLE_TOP_Z - 0.08,
    workspace_radius: float = 0.90,
) -> torch.Tensor:
    """Terminate if a cube falls off the table/workspace."""
    cube_pos = _cube_positions_w(env)
    env_origins = env.scene.env_origins.to(env.device)[:, None, :]
    local_xy = cube_pos[:, :, :2] - env_origins[:, :, :2]
    below_table = cube_pos[:, :, 2] < min_height
    outside_workspace = torch.norm(local_xy, dim=-1) > workspace_radius
    return (below_table | outside_workspace).any(dim=1)


def stack_collapsed(env: ManagerBasedRLEnv, threshold: float = 0.065, min_upright_z: float = 0.90) -> torch.Tensor:
    """Terminate when already-placed cubes drift away from the stack or tip over."""
    cube_pos = _cube_positions_w(env)
    cube_quat = _cube_orientations_w(env)
    target_pos = _target_positions_w(env)
    current_idx = _current_cube_indices(env)

    cube_ids = torch.arange(cube_pos.shape[1], device=env.device).unsqueeze(0)
    previous_cube_mask = cube_ids < current_idx.unsqueeze(1)
    distance = torch.norm(cube_pos - target_pos, dim=-1)

    qw, qx, qy, qz = cube_quat.unbind(dim=-1)
    del qw, qz
    z_axis_z = 1.0 - 2.0 * (qx.square() + qy.square())
    tilted = z_axis_z < min_upright_z

    return (previous_cube_mask & ((distance > threshold) | tilted)).any(dim=1)

