"""Reward helpers for the OpenArm 5-cube stacking task.

All reward terms are vectorized along ``num_envs`` and return a ``(num_envs,)``
tensor following Isaac Lab conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from . import observations as _obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_current_cube(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Encourages the EE to reach the currently-active cube (tanh kernel)."""
    ee = _obs._ee_position(env, ee_frame_cfg)
    cube = _obs.current_cube_position(env)
    dist = torch.norm(cube - ee, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def lifting_current_cube(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Sparse 0/1 reward for lifting the currently-active cube above ``minimal_height``."""
    cube = _obs.current_cube_position(env)
    return torch.where(cube[:, 2] > minimal_height, 1.0, 0.0)


def moving_current_cube_to_target(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Reward for moving the active (lifted) cube toward its stack target (tanh kernel)."""
    cube = _obs.current_cube_position(env)
    target = _obs.current_target_position(env)
    dist = torch.norm(target - cube, dim=-1)
    lifted = (cube[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(dist / std))


def placing_current_cube(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Fine-grained placing reward (small ``std`` tanh kernel) for the active cube."""
    cube = _obs.current_cube_position(env)
    target = _obs.current_target_position(env)
    dist = torch.norm(target - cube, dim=-1)
    lifted = (cube[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(dist / std))


def stack_success_bonus(
    env: "ManagerBasedRLEnv",
    place_threshold: float | None = None,
) -> torch.Tensor:
    """Sparse bonus that grows with the number of cubes already correctly placed.

    Returns a value in ``[0, num_cubes]`` per env.
    """
    num_cubes = int(getattr(env.cfg, "num_cubes", 5))
    thresh = float(place_threshold if place_threshold is not None else getattr(env.cfg, "place_threshold", 0.03))

    cube_pos = _obs._gather_cube_positions(env)        # (N, K, 3)
    targets = _obs._target_positions(env)              # (N, K, 3)
    xy_dist = torch.norm(cube_pos[..., :2] - targets[..., :2], dim=-1)
    z_dist = torch.abs(cube_pos[..., 2] - targets[..., 2])
    placed = (xy_dist < thresh) & (z_dist < thresh)
    # Count only **prefix** of placed cubes (so we reward order of stacking).
    cumulative = torch.cumprod(placed.to(torch.float32), dim=1)
    return cumulative.sum(dim=1)


def cube_drop_penalty(
    env: "ManagerBasedRLEnv",
    drop_height: float = -0.1,
) -> torch.Tensor:
    """Penalty (returned as a positive value to be combined with a negative weight)
    counting the number of cubes that fell below ``drop_height``.
    """
    cube_pos = _obs._gather_cube_positions(env)
    fell = (cube_pos[..., 2] < drop_height).to(torch.float32)
    return fell.sum(dim=1)
