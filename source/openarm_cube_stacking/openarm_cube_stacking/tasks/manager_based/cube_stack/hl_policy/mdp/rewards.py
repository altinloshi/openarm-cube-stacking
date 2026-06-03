from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def planner_progress(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    """Small diagnostic reward proportional to planner stage/cube progress."""
    pose_term = env.command_manager.get_term(command_name)
    stage = pose_term.planner.stage.to(torch.float32)
    cube = pose_term.planner.current_cube_idx.to(torch.float32)
    return cube + stage / 10.0
