from __future__ import annotations

"""Custom reward terms for the OpenArm LL policy.

orientation_command_error_tanh — tanh-kernel EE orientation tracking reward.
gripper_command_tracking       — soft reward for matching the gripper command.

The coarse position/orientation L2 penalties and the fine tanh position reward
are re-exported from the Isaac Lab reach-task MDP (see ``mdp/__init__.py``).
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_mul

from .observations import OPENARM_GRIPPER_OPEN_VAL

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward EE orientation tracking through ``1 - tanh(error / std)``.

    ``std`` is the bandwidth in radians (~0.1–0.2 for fine tracking).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    curr_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]

    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1.0 - torch.tanh(error / std)


def gripper_command_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    open_val: float = OPENARM_GRIPPER_OPEN_VAL,
) -> torch.Tensor:
    """Soft reward for matching the gripper opening to the commanded target.

    ``grip_cmd`` uses 0 = open, 1 = close. The current normalised opening
    fraction (1 = open, 0 = closed) is compared against the target fraction and
    mapped through ``1 - tanh(error / 0.2)``.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    finger_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    current_fraction = finger_pos.mean(dim=-1) / open_val

    grip_target = env.command_manager.get_command(command_name).squeeze(-1)
    target_fraction = 1.0 - grip_target

    error = (current_fraction - target_fraction).abs()
    return 1.0 - torch.tanh(error / 0.2)
