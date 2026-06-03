from __future__ import annotations

"""HL observation terms.

The HL classical-play / eval environments deliberately reuse the **exact** LL
observation layout so the frozen LL checkpoint runs unchanged. The LL observation
helpers (``ee_pose_in_robot_base``, ``gripper_pos_normalized``,
``grip_command_obs``) are re-exported through this package's ``mdp`` namespace.

This module adds light-weight diagnostic observations (planner stage and current
cube index) that are useful for logging but are not part of the LL policy input.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def planner_stage(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Current planner stage per env as a ``(N, 1)`` float observation."""
    term = env.command_manager.get_term(command_name)
    return term.planner.stage.to(torch.float32).unsqueeze(-1)


def current_cube_index(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Current cube index being handled per env as a ``(N, 1)`` float observation."""
    term = env.command_manager.get_term(command_name)
    return term.planner.current_cube_idx.to(torch.float32).unsqueeze(-1)
