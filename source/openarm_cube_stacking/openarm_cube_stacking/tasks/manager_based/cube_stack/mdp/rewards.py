"""Reward functions for the 5-cube stacking task.

All functions follow the Isaac Lab reward-term signature:
    func(env: ManagerBasedRLEnv, **params) -> torch.Tensor[num_envs]

Positive values encourage desired behaviour; negative values penalise
undesired behaviour.  The ManagerBasedRLEnv multiplies each function's
output by the `weight` specified in RewardTermCfg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    CUBE_NAMES,
    CUBE_SIZE,
    NUM_CUBES,
    PLACE_THRESHOLD,
    _stack_target_w,
    compute_current_cube_idx,
)


# ---------------------------------------------------------------------------
# Task-specific rewards
# ---------------------------------------------------------------------------


def reaching_current_cube(env: "ManagerBasedRLEnv", std: float = 0.1) -> torch.Tensor:
    """Exponential reward for moving the end-effector close to the current cube.

    r = exp(-dist / std)
    """
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)
    ee_pos_w = env.scene["ee_frame"].data.target_pos_w[..., 0, :]  # [N, 3]

    cube_pos_w = torch.zeros_like(ee_pos_w)
    for i in range(NUM_CUBES):
        mask = idx == i
        if mask.any():
            cube_pos_w[mask] = env.scene[CUBE_NAMES[i]].data.root_pos_w[mask]

    dist = torch.norm(cube_pos_w - ee_pos_w, dim=-1)  # [N]
    return 1.0 - torch.tanh(dist / std)


def lifting_current_cube(
    env: "ManagerBasedRLEnv", minimal_height: float = CUBE_SIZE
) -> torch.Tensor:
    """Binary reward when the current cube is lifted above the table surface.

    Returns 1.0 for envs where the current cube's z-position is above
    ``table_surface + minimal_height`` (in world frame).
    Table surface ≈ z = 0.03 (env origin + table offset).  We use 0.0 as the
    env-origin level and count anything above ``minimal_height`` as lifted.
    """
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)

    cube_pos_z = torch.zeros(env.num_envs, device=env.device)
    for i in range(NUM_CUBES):
        mask = idx == i
        if mask.any():
            cube_pos_z[mask] = env.scene[CUBE_NAMES[i]].data.root_pos_w[mask, 2]

    # env.scene.env_origins[:, 2] is the vertical offset of each env
    lifted = (cube_pos_z - env.scene.env_origins[:, 2]) > (0.04 + minimal_height)
    return lifted.float()


def moving_current_cube_to_target(
    env: "ManagerBasedRLEnv", std: float = 0.2
) -> torch.Tensor:
    """Exponential reward for moving the current cube toward its stack target.

    Only non-zero when the cube is lifted off the table.
    """
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)
    target_w = _stack_target_w(env, idx)  # [N, 3]

    cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    for i in range(NUM_CUBES):
        mask = idx == i
        if mask.any():
            cube_pos_w[mask] = env.scene[CUBE_NAMES[i]].data.root_pos_w[mask]

    dist = torch.norm(cube_pos_w - target_w, dim=-1)  # [N]
    reward = 1.0 - torch.tanh(dist / std)

    # Scale by whether the cube is lifted (avoid rewarding sliding on table)
    z_local = cube_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = (z_local > 0.04 + CUBE_SIZE).float()
    return reward * lifted


def placing_current_cube(
    env: "ManagerBasedRLEnv", std: float = 0.05
) -> torch.Tensor:
    """Fine-grained exponential reward as the cube approaches its exact target.

    Uses a tighter std than ``moving_current_cube_to_target`` to encourage
    precise placement.
    """
    idx = compute_current_cube_idx(env).clamp(max=NUM_CUBES - 1)
    target_w = _stack_target_w(env, idx)

    cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    for i in range(NUM_CUBES):
        mask = idx == i
        if mask.any():
            cube_pos_w[mask] = env.scene[CUBE_NAMES[i]].data.root_pos_w[mask]

    dist = torch.norm(cube_pos_w - target_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def cube_placed_bonus(
    env: "ManagerBasedRLEnv", threshold: float = PLACE_THRESHOLD
) -> torch.Tensor:
    """Sparse bonus awarded once per step when *any* cube enters its target zone.

    Rewards the policy every step a cube is within threshold of its target,
    acting as a per-timestep progress signal.
    """
    bonus = torch.zeros(env.num_envs, device=env.device)
    for i in range(NUM_CUBES):
        target_i = _stack_target_w(
            env, torch.full((env.num_envs,), i, dtype=torch.long, device=env.device)
        )
        cube_pos = env.scene[CUBE_NAMES[i]].data.root_pos_w
        at_target = torch.norm(cube_pos - target_i, dim=-1) < threshold
        bonus += at_target.float()
    return bonus


def stack_success_bonus(
    env: "ManagerBasedRLEnv", threshold: float = PLACE_THRESHOLD
) -> torch.Tensor:
    """Large sparse bonus when all 5 cubes are at their respective targets."""
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for i in range(NUM_CUBES):
        target_i = _stack_target_w(
            env, torch.full((env.num_envs,), i, dtype=torch.long, device=env.device)
        )
        cube_pos = env.scene[CUBE_NAMES[i]].data.root_pos_w
        at_target = torch.norm(cube_pos - target_i, dim=-1) < threshold
        all_placed &= at_target
    return all_placed.float()


def cube_drop_penalty(
    env: "ManagerBasedRLEnv", min_height: float = -0.05
) -> torch.Tensor:
    """Penalty for any cube falling below the workspace floor.

    ``min_height`` is relative to the environment origin.
    """
    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in CUBE_NAMES:
        cube_z = env.scene[name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        penalty += (cube_z < min_height).float()
    return penalty
