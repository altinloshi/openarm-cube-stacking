from __future__ import annotations

"""Custom observation terms for the OpenArm LL policy.

ee_pose_in_robot_base   — current EE (``openarm_hand``) pose in robot base frame (7D)
gripper_pos_normalized  — normalised finger opening fraction in [0, 1]           (1D)
grip_command_obs        — binary grip target from :class:`GripperCommand`         (1D)
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# OpenArm finger joints span [0.0 (closed), 0.044 (open)] m.
OPENARM_GRIPPER_OPEN_VAL: float = 0.044


def ee_pose_in_robot_base(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current end-effector pose expressed in the robot base frame.

    Returns ``(N, 7)``: ``[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]``.
    Uses the ``openarm_hand`` body so it shares the same reference frame as the
    ``UniformPoseCommand`` target, letting the policy compute tracking error
    directly from the two tensors.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_id: int = asset_cfg.body_ids[0]

    ee_pos_w = robot.data.body_pos_w[:, body_id]
    ee_quat_w = robot.data.body_quat_w[:, body_id]

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def gripper_pos_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_val: float = OPENARM_GRIPPER_OPEN_VAL,
) -> torch.Tensor:
    """Normalised mean finger opening fraction in [0, 1] (1 = open, 0 = closed)."""
    robot: Articulation = env.scene[asset_cfg.name]
    finger_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    mean_opening = finger_pos.mean(dim=-1, keepdim=True)
    return (mean_opening / open_val).clamp(0.0, 1.0)


def grip_command_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Binary gripper target from :class:`GripperCommand` (``(N, 1)``)."""
    return env.command_manager.get_command(command_name)
