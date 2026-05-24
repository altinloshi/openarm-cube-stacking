"""Reset/event helpers for the OpenArm cube stacking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import CUBE_SIZE, STACK_BASE_DEFAULT, TABLE_TOP_Z, _ensure_task_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def reset_robot_to_default(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Reset the robot articulation state and position targets."""

    env_ids = _resolve_env_ids(env, env_ids)
    robot = env.scene["robot"]

    root_state = robot.data.default_root_state[env_ids].clone()
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.reset(env_ids=env_ids)


def reset_stack_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    xy_noise: float = 0.015,
) -> None:
    """Reset the stack base target position on the table."""

    env_ids = _resolve_env_ids(env, env_ids)
    state = _ensure_task_state(env)

    stack_base = torch.tensor(STACK_BASE_DEFAULT, device=env.device, dtype=torch.float32).repeat(len(env_ids), 1)
    if xy_noise > 0.0:
        stack_base[:, :2] += (torch.rand((len(env_ids), 2), device=env.device) * 2.0 - 1.0) * xy_noise

    state["stack_base_pos"][env_ids] = stack_base
    state["best_stacked_count"][env_ids] = 0


def reset_cubes_non_overlapping(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    xy_noise: float = 0.01,
) -> None:
    """Reset cubes to separated tabletop spawn positions."""

    env_ids = _resolve_env_ids(env, env_ids)
    base_positions = torch.tensor(
        [
            [0.42, -0.16, TABLE_TOP_Z + 0.5 * CUBE_SIZE],
            [0.48, -0.05, TABLE_TOP_Z + 0.5 * CUBE_SIZE],
            [0.56, 0.14, TABLE_TOP_Z + 0.5 * CUBE_SIZE],
            [0.64, 0.02, TABLE_TOP_Z + 0.5 * CUBE_SIZE],
            [0.68, -0.12, TABLE_TOP_Z + 0.5 * CUBE_SIZE],
        ],
        device=env.device,
        dtype=torch.float32,
    )
    spawn_positions = base_positions.unsqueeze(0).repeat(len(env_ids), 1, 1)
    if xy_noise > 0.0:
        spawn_positions[:, :, :2] += (torch.rand((len(env_ids), 5, 2), device=env.device) * 2.0 - 1.0) * xy_noise

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat(len(env_ids), 1)
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device, dtype=torch.float32)

    for cube_index in range(5):
        cube = env.scene[f"cube_{cube_index}"]
        root_pose = torch.cat((spawn_positions[:, cube_index, :], identity_quat), dim=-1)
        cube.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        cube.reset(env_ids=env_ids)
