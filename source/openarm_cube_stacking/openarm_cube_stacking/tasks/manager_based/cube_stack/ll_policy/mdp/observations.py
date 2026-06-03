from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from ...tabletop_scene import OPENARM_GRIPPER_OPEN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_in_robot_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Current EE pose in the OpenArm root frame as ``[pos, quat_wxyz]``."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def gripper_pos_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_val: float = OPENARM_GRIPPER_OPEN,
) -> torch.Tensor:
    """Mean OpenArm finger opening normalized to [0, 1]."""
    robot: Articulation = env.scene[asset_cfg.name]
    finger_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return (finger_pos.mean(dim=-1, keepdim=True) / open_val).clamp(0.0, 1.0)


def grip_command_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return binary commanded grip state: 0=open, 1=close."""
    return env.command_manager.get_command(command_name)
