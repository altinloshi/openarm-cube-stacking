"""Termination terms for the HL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...openarm_lift_style_scene_cfg import CUBE_NAMES, CUBE_SIZE, TABLE_TOP_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when episode time limit is reached."""
    return env.episode_length_buf >= env.max_episode_length


def all_cubes_stacked(
    env: ManagerBasedRLEnv,
    threshold: float = 0.04,
) -> torch.Tensor:
    """Terminate when all cubes reach their sequential stack targets."""
    from .rewards import _cube_positions, _stack_target_positions

    dist = torch.norm(_cube_positions(env) - _stack_target_positions(env), dim=-1)
    return (dist < threshold).all(dim=-1)


def planner_done(env: ManagerBasedRLEnv, command_name: str = "ee_pose") -> torch.Tensor:
    """Terminate when the classical planner reaches the DONE stage."""
    from ..classical_stack_planner import PlannerStage

    cmd_mgr = getattr(env, "command_manager", None)
    if cmd_mgr is not None and cmd_mgr.has_term(command_name):
        cmd_term = cmd_mgr.get_term(command_name)
        if hasattr(cmd_term, "_planner"):
            return cmd_term._planner.stage == PlannerStage.DONE
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
