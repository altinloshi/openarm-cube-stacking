"""Observation functions for the 5-cube stacking task.

All functions follow the Isaac Lab observation-term signature:
    func(env: ManagerBasedRLEnv, **params) -> torch.Tensor

Tensors are 2-D: [num_envs, obs_dim].  The observation manager concatenates
them along dim=-1 when `concatenate_terms=True`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Task constants (kept here to avoid a circular import from the cfg module)
# ---------------------------------------------------------------------------

NUM_CUBES: int = 5
CUBE_NAMES: list[str] = [f"cube_{i}" for i in range(NUM_CUBES)]
CUBE_SIZE: float = 0.05

# Stack base in env-local coordinates (matches cube_stack_env_cfg.py)
STACK_BASE_LOCAL_X: float = 0.55
STACK_BASE_LOCAL_Y: float = 0.0
STACK_BASE_LOCAL_Z: float = 0.055  # centre of cube_0 when placed on table

# Threshold (m) to consider a cube "placed" at its stack target
PLACE_THRESHOLD: float = 0.03


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stack_target_w(env: "ManagerBasedRLEnv", cube_idx: torch.Tensor) -> torch.Tensor:
    """Compute the world-frame stack target for each env given per-env cube indices.

    Args:
        env: The RL environment.
        cube_idx: Long tensor of shape [num_envs] with cube index in [0, NUM_CUBES-1].

    Returns:
        Tensor of shape [num_envs, 3] with world-frame target positions.
    """
    target = env.scene.env_origins.clone()  # [num_envs, 3]
    target[:, 0] += STACK_BASE_LOCAL_X
    target[:, 1] += STACK_BASE_LOCAL_Y
    target[:, 2] += STACK_BASE_LOCAL_Z + cube_idx.float() * CUBE_SIZE
    return target


def compute_current_cube_idx(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return the index of the first unplaced cube for each environment.

    A cube is considered "placed" when it is within PLACE_THRESHOLD of its
    stack target AND all lower-index cubes are also placed (sequential order).

    Returns:
        Long tensor of shape [num_envs] with values in [0, NUM_CUBES].
        A value of NUM_CUBES means all cubes are placed (success state).
    """
    # placed[:, i] is True when cube i is close to its target position
    placed = torch.zeros(
        env.num_envs, NUM_CUBES, dtype=torch.bool, device=env.device
    )
    for i in range(NUM_CUBES):
        target_i = _stack_target_w(
            env, torch.full((env.num_envs,), i, dtype=torch.long, device=env.device)
        )
        cube_pos = env.scene[CUBE_NAMES[i]].data.root_pos_w  # [num_envs, 3]
        placed[:, i] = torch.norm(cube_pos - target_i, dim=-1) < PLACE_THRESHOLD

    # Current index = length of the leading run of True values.
    # Example: [T, T, F, F, F] → current_idx = 2 (cube_2 is next).
    # cumprod gives [1, 1, 0, 0, 0] → sum = 2.
    current_idx = placed.float().cumprod(dim=1).sum(dim=1).long()  # [num_envs]
    return current_idx  # 0 … NUM_CUBES


# ---------------------------------------------------------------------------
# Observation term functions
# ---------------------------------------------------------------------------


def ee_position_in_env_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """End-effector position relative to the env origin, shape [num_envs, 3]."""
    ee_pos_w = env.scene["ee_frame"].data.target_pos_w[..., 0, :]  # [N, 3]
    return ee_pos_w - env.scene.env_origins


def cube_positions_in_env_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """All 5 cube positions in env-local frame, flattened to [num_envs, 15]."""
    parts = []
    for name in CUBE_NAMES:
        pos_w = env.scene[name].data.root_pos_w  # [N, 3]
        parts.append(pos_w - env.scene.env_origins)
    return torch.cat(parts, dim=-1)  # [N, 15]


def cube_orientations(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """All 5 cube orientations (quaternions wxyz), flattened to [num_envs, 20]."""
    parts = []
    for name in CUBE_NAMES:
        quat = env.scene[name].data.root_quat_w  # [N, 4]
        parts.append(quat)
    return torch.cat(parts, dim=-1)  # [N, 20]


def current_cube_index_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Normalised current cube index, shape [num_envs, 1].

    Returns the index clamped to [0, NUM_CUBES-1] and normalised to [0, 1].
    """
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)  # [N]
    return (idx.float() / (NUM_CUBES - 1)).unsqueeze(-1)  # [N, 1]


def current_target_position_in_env_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Stack target position for the current cube, in env-local frame.

    Shape: [num_envs, 3].
    """
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)
    target_w = _stack_target_w(env, idx)  # [N, 3]
    return target_w - env.scene.env_origins


def ee_to_current_cube(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vector from the end-effector to the current cube, shape [num_envs, 3]."""
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)  # [N]
    ee_pos_w = env.scene["ee_frame"].data.target_pos_w[..., 0, :]  # [N, 3]

    # Gather the position of the current cube for each env
    cube_pos_w = torch.zeros_like(ee_pos_w)
    for i in range(NUM_CUBES):
        mask = idx == i  # [N]
        if mask.any():
            cube_pos_w[mask] = env.scene[CUBE_NAMES[i]].data.root_pos_w[mask]

    return cube_pos_w - ee_pos_w  # vector EE → cube


def current_cube_to_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vector from the current cube to its stack target, shape [num_envs, 3]."""
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)
    target_w = _stack_target_w(env, idx)  # [N, 3]

    cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    for i in range(NUM_CUBES):
        mask = idx == i
        if mask.any():
            cube_pos_w[mask] = env.scene[CUBE_NAMES[i]].data.root_pos_w[mask]

    return target_w - cube_pos_w  # vector cube → target
