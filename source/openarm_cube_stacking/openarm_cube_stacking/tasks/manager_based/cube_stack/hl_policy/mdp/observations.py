"""Observation terms for the HL policy environment.

The HL-play/eval environments use the SAME policy-level observations as the LL
environment so that a frozen LL policy can be applied without modification.

Additional observation groups expose cube positions for evaluation/metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

# Re-use all LL observation terms
from ...ll_policy.mdp.observations import (  # noqa: F401
    GRIPPER_OPEN_LIMIT,
    _get_ee_pose_w,
    _get_robot_base_pose_w,
    ee_pos_b,
    ee_quat_b,
    gripper_opening_norm,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    target_ee_pose_command,
    target_gripper_cmd,
)
from ...openarm_lift_style_scene_cfg import CUBE_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────────────────────────────────────
# Additional cube-level observations (for HL monitoring, not fed to LL policy)
# ─────────────────────────────────────────────────────────────────────────────


def cube_positions_w(
    env: "ManagerBasedRLEnv",
    cube_names: Sequence[str] = CUBE_NAMES,
) -> torch.Tensor:
    """All cube positions in world frame, shape (num_envs, num_cubes * 3)."""
    positions = [env.scene[name].data.root_pos_w for name in cube_names]
    return torch.stack(positions, dim=1).reshape(env.num_envs, -1)


def cube_orientations_w(
    env: "ManagerBasedRLEnv",
    cube_names: Sequence[str] = CUBE_NAMES,
) -> torch.Tensor:
    """All cube orientations in world frame, shape (num_envs, num_cubes * 4)."""
    orientations = [env.scene[name].data.root_quat_w for name in cube_names]
    return torch.stack(orientations, dim=1).reshape(env.num_envs, -1)


def planner_stage(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    """Current planner stage for each env, shape (num_envs, 1)."""
    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term, "_planner"):
        return cmd_term._planner.stage.unsqueeze(-1).float()
    return torch.zeros((env.num_envs, 1), device=env.device)


def planner_cube_idx(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    """Current target cube index for each env, shape (num_envs, 1)."""
    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term, "_planner"):
        return cmd_term._planner.cube_idx.unsqueeze(-1).float()
    return torch.zeros((env.num_envs, 1), device=env.device)
