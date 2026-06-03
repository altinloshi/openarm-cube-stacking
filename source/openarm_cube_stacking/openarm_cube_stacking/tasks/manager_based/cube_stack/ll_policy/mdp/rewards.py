from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_mul

from ...tabletop_scene import OPENARM_GRIPPER_OPEN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Fine orientation tracking reward for the commanded EE quaternion."""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired_quat_w = quat_mul(robot.data.root_quat_w, command[:, 3:7])
    current_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    error = quat_error_magnitude(current_quat_w, desired_quat_w)
    return 1.0 - torch.tanh(error / std)


def gripper_command_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    open_val: float = OPENARM_GRIPPER_OPEN,
) -> torch.Tensor:
    """Reward matching the binary gripper command."""
    robot: Articulation = env.scene[asset_cfg.name]
    finger_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    current_open_fraction = (finger_pos.mean(dim=-1) / open_val).clamp(0.0, 1.0)
    grip_target = env.command_manager.get_command(command_name).squeeze(-1)
    target_open_fraction = 1.0 - grip_target
    return 1.0 - torch.tanh((current_open_fraction - target_open_fraction).abs() / 0.2)
