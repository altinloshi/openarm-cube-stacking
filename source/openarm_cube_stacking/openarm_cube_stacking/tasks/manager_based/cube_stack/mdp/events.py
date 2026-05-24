"""Event helpers used for reset/randomization in OpenArm cube stacking."""

from __future__ import annotations

import torch

from . import observations as obs


def _resolve_env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def reset_stack_target(env, env_ids: torch.Tensor | None = None, xy_noise: float = 0.0) -> None:
    """Reset stack base target; randomization is optional for early curriculum."""
    env_ids = _resolve_env_ids(env, env_ids)
    base = torch.tensor(obs.STACK_BASE_DEFAULT, device=env.device, dtype=torch.float32).repeat(env_ids.numel(), 1)
    if xy_noise > 0.0:
        base[:, :2] += torch.empty((env_ids.numel(), 2), device=env.device).uniform_(-xy_noise, xy_noise)

    if not hasattr(env, "stack_target_base"):
        full_base = torch.tensor(obs.STACK_BASE_DEFAULT, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        env.stack_target_base = full_base
    env.stack_target_base[env_ids] = base


def reset_robot_to_default(
    env,
    env_ids: torch.Tensor | None = None,
    asset_name: str = "robot",
    joint_pos_noise: float = 0.04,
) -> None:
    """Reset robot joints near defaults."""
    env_ids = _resolve_env_ids(env, env_ids)
    robot = env.scene[asset_name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    if joint_pos_noise > 0.0:
        noise = torch.empty_like(joint_pos).uniform_(-joint_pos_noise, joint_pos_noise)
        joint_pos += noise
        if hasattr(robot.data, "soft_joint_pos_limits"):
            lower = robot.data.soft_joint_pos_limits[env_ids, :, 0]
            upper = robot.data.soft_joint_pos_limits[env_ids, :, 1]
            joint_pos = torch.clamp(joint_pos, min=lower, max=upper)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_cubes_non_overlapping(env, env_ids: torch.Tensor | None = None, xy_noise: float = 0.01) -> None:
    """Reset cubes to deterministic, non-overlapping table positions."""
    env_ids = _resolve_env_ids(env, env_ids)
    num_ids = env_ids.numel()

    base_positions = torch.tensor(
        [
            [0.44, -0.14, obs.TABLE_TOP_Z + 0.5 * obs.CUBE_SIZE],
            [0.44, -0.06, obs.TABLE_TOP_Z + 0.5 * obs.CUBE_SIZE],
            [0.44, 0.02, obs.TABLE_TOP_Z + 0.5 * obs.CUBE_SIZE],
            [0.52, -0.10, obs.TABLE_TOP_Z + 0.5 * obs.CUBE_SIZE],
            [0.52, -0.02, obs.TABLE_TOP_Z + 0.5 * obs.CUBE_SIZE],
        ],
        device=env.device,
        dtype=torch.float32,
    )
    positions = base_positions.unsqueeze(0).repeat(num_ids, 1, 1)
    if xy_noise > 0.0:
        positions[:, :, :2] += torch.empty((num_ids, obs.NUM_CUBES, 2), device=env.device).uniform_(-xy_noise, xy_noise)

    env_origins = env.scene.env_origins[env_ids]
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat(num_ids, 1)

    for cube_i, cube_name in enumerate(obs.CUBE_NAMES):
        cube = env.scene[cube_name]
        root_state = cube.data.default_root_state[env_ids].clone()
        root_state[:, :3] = positions[:, cube_i, :] + env_origins
        root_state[:, 3:7] = identity_quat
        root_state[:, 7:] = 0.0
        cube.write_root_state_to_sim(root_state, env_ids=env_ids)
