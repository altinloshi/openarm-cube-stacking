from __future__ import annotations

"""Reward terms for the HL cube-stacking environment.

The HL classical-play / eval environments are executed by the frozen LL policy
(no HL learning), so rewards are optional and used only for logging / analysis.
``stack_progress`` reports the fraction of cubes currently stacked.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ...tabletop import CUBE_NAMES
from .terminations import _cube_states, cube_stacked_mask, stack_target_positions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stack_progress(
    env: ManagerBasedRLEnv,
    cube_names: Sequence[str] = CUBE_NAMES,
    pose_cmd_name: str = "ee_pose",
    xy_tol: float = 0.03,
    z_tol: float = 0.025,
    vel_tol: float = 0.05,
) -> torch.Tensor:
    """Fraction of cubes currently stacked and stable, in [0, 1]."""
    cube_pos, cube_vel = _cube_states(env, cube_names)
    target_pos = stack_target_positions(env, pose_cmd_name, len(cube_names))
    placed = cube_stacked_mask(cube_pos, cube_vel, target_pos, xy_tol, z_tol, vel_tol)
    return placed.float().mean(dim=1)
