from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pose_term(env: "ManagerBasedRLEnv", command_name: str = "ee_pose"):
    return env.command_manager.get_term(command_name)


def planner_stage(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    return _pose_term(env, command_name).planner.stage.to(torch.float32).unsqueeze(-1)


def current_cube_index(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    return _pose_term(env, command_name).planner.current_cube_idx.to(torch.float32).unsqueeze(-1)


def planner_target_pose(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    pose_term = _pose_term(env, command_name)
    return torch.cat([pose_term._target_pos_w, pose_term._target_quat_w], dim=-1)


def stack_base_position(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    return _pose_term(env, command_name).stack_base_pos_w
