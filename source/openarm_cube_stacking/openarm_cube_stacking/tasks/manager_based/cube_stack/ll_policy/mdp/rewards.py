# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the LL policy (EE pose tracking + gripper tracking).

All rewards are shaped (num_envs,) and meant to be weighted in RewardsCfg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms

from .observations import (
    _get_ee_pose_w,
    _get_robot_base_pose_w,
    GRIPPER_OPEN_LIMIT,
    target_gripper_cmd,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_target_ee_pose_w(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return commanded target EE pose in world frame."""
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 7)
    target_pos_b = cmd[:, :3]
    target_quat_b = cmd[:, 3:7]
    base_pos_w, base_quat_w = _get_robot_base_pose_w(env, asset_cfg)
    # Transform target from robot base frame to world frame
    from isaaclab.utils.math import combine_frame_transforms
    target_pos_w, target_quat_w = combine_frame_transforms(
        base_pos_w, base_quat_w, target_pos_b, target_quat_b
    )
    return target_pos_w, target_quat_w


# ─────────────────────────────────────────────────────────────────────────────
# EE position tracking rewards
# ─────────────────────────────────────────────────────────────────────────────


def ee_position_tracking_coarse(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    std: float = 0.2,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense penalty based on coarse EE position error (tanh kernel, large std)."""
    ee_pos_w, _ = _get_ee_pose_w(env, ee_frame_cfg)
    target_pos_w, _ = _get_target_ee_pose_w(env, command_name, asset_cfg)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def ee_position_tracking_fine(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    std: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for fine EE position tracking (tanh kernel, small std)."""
    ee_pos_w, _ = _get_ee_pose_w(env, ee_frame_cfg)
    target_pos_w, _ = _get_target_ee_pose_w(env, command_name, asset_cfg)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


# ─────────────────────────────────────────────────────────────────────────────
# EE orientation tracking rewards
# ─────────────────────────────────────────────────────────────────────────────


def ee_orientation_tracking_coarse(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    std: float = 0.5,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense penalty based on coarse EE orientation error (tanh kernel, large std)."""
    _, ee_quat_w = _get_ee_pose_w(env, ee_frame_cfg)
    _, target_quat_w = _get_target_ee_pose_w(env, command_name, asset_cfg)
    ori_err = quat_error_magnitude(ee_quat_w, target_quat_w)  # radians
    return 1.0 - torch.tanh(ori_err / std)


def ee_orientation_tracking_fine(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    std: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for fine EE orientation tracking (tanh kernel, small std)."""
    _, ee_quat_w = _get_ee_pose_w(env, ee_frame_cfg)
    _, target_quat_w = _get_target_ee_pose_w(env, command_name, asset_cfg)
    ori_err = quat_error_magnitude(ee_quat_w, target_quat_w)
    return 1.0 - torch.tanh(ori_err / std)


# ─────────────────────────────────────────────────────────────────────────────
# Gripper tracking reward
# ─────────────────────────────────────────────────────────────────────────────


def gripper_command_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary reward for matching gripper state to commanded state.

    Returns 1.0 when gripper opening direction matches the commanded open/close.
    """
    from .observations import gripper_opening_norm

    current_open = gripper_opening_norm(env, asset_cfg).squeeze(-1)  # [0,1]
    desired_open = target_gripper_cmd(env).squeeze(-1)               # 0 or 1
    # Reward as 1 - |desired - current|
    return 1.0 - torch.abs(desired_open - current_open)


# ─────────────────────────────────────────────────────────────────────────────
# Regularisation penalties
# ─────────────────────────────────────────────────────────────────────────────


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise action change rate: -||a_t - a_{t-1}||^2."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=-1,
    )


def joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise robot joint velocities: -||joint_vel||^2."""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel), dim=-1)
