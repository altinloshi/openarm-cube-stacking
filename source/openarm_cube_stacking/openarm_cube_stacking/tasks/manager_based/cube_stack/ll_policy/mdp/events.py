# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the LL policy environment.

The LL task only needs the robot reset and the gripper-command reset.
Cube resets are handled by the HL/Eval layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(device=env.device)


def reset_robot_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the OpenArm root pose and joint state to defaults."""
    env_ids = _resolve_env_ids(env, env_ids)
    robot: Articulation = env.scene[asset_cfg.name]

    root_state = robot.data.default_root_state[env_ids].clone()
    # default_root_state stores LOCAL position; add env origins to get world pos
    root_state[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.reset(env_ids)


def reset_gripper_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    open_probability: float = 0.5,
) -> None:
    """Sample a binary gripper command once per episode.

    Stores result in ``env.gripper_cmd`` with shape ``(num_envs, 1)``,
    where 1.0 = open and 0.0 = close.
    """
    env_ids = _resolve_env_ids(env, env_ids)

    if not hasattr(env, "gripper_cmd") or env.gripper_cmd.shape[0] != env.num_envs:
        env.gripper_cmd = torch.zeros((env.num_envs, 1), device=env.device)

    sampled = (torch.rand(len(env_ids), 1, device=env.device) < open_probability).float()
    env.gripper_cmd[env_ids] = sampled
