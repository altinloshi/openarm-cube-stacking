from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .observations import CUBE_NAMES, DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS, DEFAULT_STACK_BASE_LOCAL_POS

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
    """Reset the OpenArm root and joints to the configured default state."""
    env_ids = _resolve_env_ids(env, env_ids)
    robot: Articulation = env.scene[asset_cfg.name]

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.reset(env_ids)


def reset_cubes_non_overlapping(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    cube_names: Sequence[str] = CUBE_NAMES,
    local_positions: Sequence[Sequence[float]] = DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    position_noise: float = 0.015,
) -> None:
    """Reset all cubes to non-overlapping positions on the table.

    The spawn layout is deterministic plus optional per-reset XY jitter. Keeping the reset helper
    centralized makes it straightforward to replace this with richer sampling or curricula later.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    origins = env.scene.env_origins[env_ids]
    local_pos = torch.tensor(local_positions, dtype=torch.float32, device=env.device)

    if position_noise > 0.0:
        noise = torch.empty((len(env_ids), len(cube_names), 2), device=env.device).uniform_(-position_noise, position_noise)
    else:
        noise = torch.zeros((len(env_ids), len(cube_names), 2), device=env.device)

    quat = torch.zeros((len(env_ids), 4), device=env.device)
    quat[:, 0] = 1.0
    velocity = torch.zeros((len(env_ids), 6), device=env.device)

    for cube_id, cube_name in enumerate(cube_names):
        cube: RigidObject = env.scene[cube_name]
        pos = origins + local_pos[cube_id].unsqueeze(0)
        pos[:, :2] += noise[:, cube_id, :]
        cube.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        cube.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        cube.reset(env_ids)


def reset_stack_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    local_stack_base: Sequence[float] = DEFAULT_STACK_BASE_LOCAL_POS,
    position_noise: float = 0.0,
) -> None:
    """Reset the per-environment stack base position in world frame."""
    env_ids = _resolve_env_ids(env, env_ids)
    if not hasattr(env, "stack_base_pos_w") or env.stack_base_pos_w.shape[0] != env.num_envs:
        env.stack_base_pos_w = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    local_pos = torch.tensor(local_stack_base, dtype=torch.float32, device=env.device).repeat(len(env_ids), 1)
    if position_noise > 0.0:
        local_pos[:, :2] += torch.empty((len(env_ids), 2), device=env.device).uniform_(-position_noise, position_noise)

    env.stack_base_pos_w[env_ids] = env.scene.env_origins[env_ids] + local_pos

