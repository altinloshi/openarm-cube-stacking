# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the HL play/eval environment.

In HL mode the LL policy executes the motion; these rewards measure
stacking progress for evaluation/monitoring, not for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...openarm_lift_style_scene_cfg import (
    CUBE_NAMES,
    CUBE_SIZE,
    OPENARM_LIFT_STACK_BASE_LOCAL_POS,
    TABLE_TOP_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cube_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cube positions (num_envs, num_cubes, 3)."""
    return torch.stack(
        [env.scene[name].data.root_pos_w for name in CUBE_NAMES], dim=1
    )


def _stack_target_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target positions for each cube in the stack (num_envs, num_cubes, 3)."""
    stack_base = getattr(env, "stack_base_pos_w", None)
    if stack_base is None:
        local = torch.tensor(
            list(OPENARM_LIFT_STACK_BASE_LOCAL_POS[:3]), dtype=torch.float32, device=env.device
        )
        stack_base = env.scene.env_origins + local.unsqueeze(0)

    n = len(CUBE_NAMES)
    z_offsets = torch.arange(n, device=env.device, dtype=torch.float32) * CUBE_SIZE
    targets = stack_base[:, None, :].expand(-1, n, -1).clone()
    targets[:, :, 2] += z_offsets[None, :]
    return targets


def cubes_at_target(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.04,
) -> torch.Tensor:
    """Number of cubes within threshold of their stack target, normalised to [0, 1]."""
    dist = torch.norm(_cube_positions(env) - _stack_target_positions(env), dim=-1)
    return (dist < threshold).float().sum(dim=-1) / len(CUBE_NAMES)


def stack_success_bonus(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.04,
) -> torch.Tensor:
    """1.0 if all cubes are at their targets, else 0.0."""
    dist = torch.norm(_cube_positions(env) - _stack_target_positions(env), dim=-1)
    return (dist < threshold).all(dim=-1).float()
