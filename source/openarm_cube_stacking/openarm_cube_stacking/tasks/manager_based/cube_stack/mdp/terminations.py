"""Termination functions for the 5-cube stacking task.

All functions follow the Isaac Lab termination-term signature:
    func(env: ManagerBasedRLEnv, **params) -> torch.Tensor[bool, num_envs]

A True value triggers episode termination for that environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import CUBE_NAMES, NUM_CUBES, PLACE_THRESHOLD, _stack_target_w


def all_cubes_stacked(
    env: "ManagerBasedRLEnv", threshold: float = PLACE_THRESHOLD
) -> torch.Tensor:
    """Return True when all 5 cubes are within threshold of their stack targets.

    This is the success termination condition.
    """
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for i in range(NUM_CUBES):
        target_i = _stack_target_w(
            env, torch.full((env.num_envs,), i, dtype=torch.long, device=env.device)
        )
        cube_pos = env.scene[CUBE_NAMES[i]].data.root_pos_w
        at_target = torch.norm(cube_pos - target_i, dim=-1) < threshold
        all_placed &= at_target
    return all_placed


def cube_dropped(
    env: "ManagerBasedRLEnv", min_height: float = -0.05
) -> torch.Tensor:
    """Return True when any cube falls below the workspace floor.

    ``min_height`` is measured relative to each env's origin, so a value of
    -0.05 means the cube fell 5 cm below the ground plane.
    """
    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in CUBE_NAMES:
        cube_z = env.scene[name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        any_dropped |= cube_z < min_height
    return any_dropped


def stack_collapsed(
    env: "ManagerBasedRLEnv",
    tilt_threshold_cos: float = 0.9659,  # cos(15°)
) -> torch.Tensor:
    """Return True when any previously placed cube has tilted excessively.

    A cube is considered tilted if its local z-axis deviates more than
    ``tilt_threshold`` from the world z-axis (detected via quaternion).
    Only checks cubes that should already be placed (lower indices).

    NOTE: This is a soft check based on orientation only; a full collapse
    check would require tracking velocities and force sensors.

    Args:
        env: The RL environment.
        tilt_threshold_cos: Cosine of the maximum allowed tilt angle.
                            Default 0.9659 ≈ cos(15°).
    """
    collapsed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in CUBE_NAMES:
        quat = env.scene[name].data.root_quat_w  # [N, 4] wxyz
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # z-component of the cube's local z-axis expressed in world frame
        z_world_z = 1.0 - 2.0 * (qx**2 + qy**2)
        collapsed |= z_world_z < tilt_threshold_cos
    return collapsed
