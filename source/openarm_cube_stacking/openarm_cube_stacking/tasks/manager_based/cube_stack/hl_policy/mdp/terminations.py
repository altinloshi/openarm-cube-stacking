from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...end_to_end.mdp.terminations import cube_dropped, stack_collapsed
from ...eval.metrics import compute_stack_success_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def planner_succeeded(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    pose_term = env.command_manager.get_term(command_name)
    return pose_term.planner.is_done() & compute_stack_success_mask(env).all(dim=1)


def planner_failed(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    pose_term = env.command_manager.get_term(command_name)
    return pose_term.planner.episode_failed
