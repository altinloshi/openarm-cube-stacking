# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the LL policy environment.

The LL policy observes:
  - Robot joint state (positions + velocities)
  - Current end-effector pose in the robot-base frame
  - Commanded target EE pose from the command manager
  - Commanded gripper state (set by reset_gripper_command event)
  - Current normalised gripper opening
  - Last action taken by the policy

All tensors are shaped (num_envs, dim).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_inv, quat_mul, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ────────────────────────────────────────────────────────────────────────────
# Gripper finger joint open limit (from OPENARM_UNI_CFG init_state)
GRIPPER_OPEN_LIMIT: float = 0.044


# ────────────────────────────────────────────────────────────────────────────
# Robot joint state
# ────────────────────────────────────────────────────────────────────────────


def joint_pos_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions relative to their defaults, shape (num_envs, n_joints)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities relative to their defaults (usually zero), shape (num_envs, n_joints)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel - robot.data.default_joint_vel


# ────────────────────────────────────────────────────────────────────────────
# End-effector state
# ────────────────────────────────────────────────────────────────────────────


def _get_ee_pose_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ee_pos_w, ee_quat_w) in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_w = ee_frame.data.target_pos_w[:, 0, :]    # (num_envs, 3)
    quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (num_envs, 4)
    return pos_w, quat_w


def _get_robot_base_pose_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (base_pos_w, base_quat_w) for the robot root link."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w, robot.data.root_quat_w


def ee_pos_b(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """EE position expressed in the robot base frame, shape (num_envs, 3)."""
    ee_pos_w, ee_quat_w = _get_ee_pose_w(env, ee_frame_cfg)
    base_pos_w, base_quat_w = _get_robot_base_pose_w(env, asset_cfg)
    pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
    return pos_b


def ee_quat_b(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """EE orientation in the robot base frame (quaternion wxyz), shape (num_envs, 4)."""
    ee_pos_w, ee_quat_w = _get_ee_pose_w(env, ee_frame_cfg)
    base_pos_w, base_quat_w = _get_robot_base_pose_w(env, asset_cfg)
    _, quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
    return quat_b


# ────────────────────────────────────────────────────────────────────────────
# Command observations
# ────────────────────────────────────────────────────────────────────────────


def target_ee_pose_command(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Target EE pose from the command manager, shape (num_envs, 7) = pos(3)+quat(4)."""
    return env.command_manager.get_command(command_name)


def target_gripper_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary gripper command set each episode, shape (num_envs, 1).

    1.0 = open, 0.0 = close.  Initialised lazily on first access.
    """
    if not hasattr(env, "gripper_cmd") or env.gripper_cmd.shape[0] != env.num_envs:
        env.gripper_cmd = torch.zeros((env.num_envs, 1), device=env.device)
    return env.gripper_cmd


def gripper_opening_norm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalised current gripper opening [0, 1], shape (num_envs, 1).

    0 = fully closed, 1 = fully open.
    Uses the average of the two finger joint positions.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    finger_mask = robot.find_joints("openarm_finger_joint.*")[0]
    if len(finger_mask) == 0:
        return torch.zeros((env.num_envs, 1), device=env.device)
    finger_pos = robot.data.joint_pos[:, finger_mask]      # (num_envs, n_fingers)
    opening = finger_pos.mean(dim=-1, keepdim=True) / GRIPPER_OPEN_LIMIT
    return opening.clamp(0.0, 1.0)


# ────────────────────────────────────────────────────────────────────────────
# Last action
# ────────────────────────────────────────────────────────────────────────────


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last applied action, shape (num_envs, action_dim)."""
    return env.action_manager.action
