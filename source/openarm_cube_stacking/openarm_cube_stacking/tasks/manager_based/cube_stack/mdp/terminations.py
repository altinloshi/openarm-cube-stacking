"""Termination helpers for the OpenArm cube-stacking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import observations as _obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_cubes_stacked(
    env: "ManagerBasedRLEnv",
    place_threshold: float | None = None,
) -> torch.Tensor:
    """Done when all cubes are within ``place_threshold`` of their target stack slot."""
    thresh = float(place_threshold if place_threshold is not None else getattr(env.cfg, "place_threshold", 0.03))
    cube_pos = _obs._gather_cube_positions(env)
    targets = _obs._target_positions(env)
    xy_dist = torch.norm(cube_pos[..., :2] - targets[..., :2], dim=-1)
    z_dist = torch.abs(cube_pos[..., 2] - targets[..., 2])
    placed = (xy_dist < thresh) & (z_dist < thresh)
    return placed.all(dim=1)


def cube_dropped(
    env: "ManagerBasedRLEnv",
    drop_height: float = -0.1,
) -> torch.Tensor:
    """Done if **any** cube has fallen below ``drop_height``."""
    cube_pos = _obs._gather_cube_positions(env)
    return (cube_pos[..., 2] < drop_height).any(dim=1)


def stack_collapsed(
    env: "ManagerBasedRLEnv",
    tilt_cosine: float = 0.9,
) -> torch.Tensor:
    """Done if any cube has tilted more than ``acos(tilt_cosine)`` away from world-up.

    Uses the cube's body-frame Z-axis transformed to world to detect tipping.
    """
    quats = _obs._gather_cube_quats(env)  # (N, K, 4) wxyz
    qw, qx, qy, qz = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    # Z-axis of body in world frame
    z_x = 2.0 * (qx * qz + qw * qy)
    z_y = 2.0 * (qy * qz - qw * qx)
    z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    # dot with world up = z_z directly
    tilted = z_z < tilt_cosine  # (N, K)
    return tilted.any(dim=1)
