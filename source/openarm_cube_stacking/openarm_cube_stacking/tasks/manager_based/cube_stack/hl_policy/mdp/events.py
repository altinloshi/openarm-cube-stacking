from __future__ import annotations

"""Reset events for the HL cube-stacking environment.

Two reset paths are provided:

* :func:`reset_cubes_and_stack` — randomised cube spawn + stack base for the
  HL classical-play environment (reachable randomisation).
* :func:`reset_from_scenarios` — deterministic tournament scenarios for the eval
  environment (scenario index = ``env_id % num_scenarios``; no randomisation).

Both set the per-environment stack base on the :class:`HLStackPoseCommand` term
so the planner stacks at the correct location.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from ...tabletop import (
    CUBE_NAMES,
    DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    DEFAULT_STACK_BASE_LOCAL_POS,
)
from .commands import HLStackPoseCommand

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


def _write_cube_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_name: str,
    pos_w: torch.Tensor,
    yaw: torch.Tensor | None = None,
) -> None:
    cube: RigidObject = env.scene[cube_name]
    if yaw is None:
        quat = torch.zeros((len(env_ids), 4), device=env.device)
        quat[:, 0] = 1.0
    else:
        zeros = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
    velocity = torch.zeros((len(env_ids), 6), device=env.device)
    cube.write_root_pose_to_sim(torch.cat((pos_w, quat), dim=-1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    cube.reset(env_ids)


def reset_cubes_and_stack(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    cube_names: Sequence[str] = CUBE_NAMES,
    local_positions: Sequence[Sequence[float]] = DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    stack_base_local: Sequence[float] = DEFAULT_STACK_BASE_LOCAL_POS,
    cube_position_noise: float = 0.02,
    stack_position_noise: float = 0.0,
    pose_cmd_name: str = "ee_pose",
) -> None:
    """Randomised reset of cubes and stack base (HL classical-play)."""
    env_ids = _resolve_env_ids(env, env_ids)
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    local_pos = torch.tensor(local_positions, dtype=torch.float32, device=env.device)

    for cube_id, cube_name in enumerate(cube_names):
        pos = origins + local_pos[cube_id].unsqueeze(0)
        if cube_position_noise > 0.0:
            pos[:, :2] += torch.empty((n, 2), device=env.device).uniform_(-cube_position_noise, cube_position_noise)
        _write_cube_pose(env, env_ids, cube_name, pos)

    base = torch.tensor(stack_base_local, dtype=torch.float32, device=env.device).repeat(n, 1)
    if stack_position_noise > 0.0:
        base[:, :2] += torch.empty((n, 2), device=env.device).uniform_(-stack_position_noise, stack_position_noise)
    pose_term: HLStackPoseCommand = env.command_manager.get_term(pose_cmd_name)
    pose_term.set_stack_base(env_ids, base)


def reset_from_scenarios(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    cube_names: Sequence[str] = CUBE_NAMES,
    pose_cmd_name: str = "ee_pose",
) -> None:
    """Deterministic reset from the 30 tournament scenarios (eval).

    Scenario index for each env is ``env_id % num_scenarios``. Cube positions,
    optional cube yaw and the stack base are read from :mod:`eval.scenarios`.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    from ...eval.scenarios import get_scenario_tensors

    cube_pos_local, cube_yaw, stack_base_local = get_scenario_tensors(env.device)
    num_scenarios = cube_pos_local.shape[0]
    scenario_idx = (env_ids % num_scenarios).to(torch.long)

    origins = env.scene.env_origins[env_ids]
    for cube_id, cube_name in enumerate(cube_names):
        local = cube_pos_local[scenario_idx, cube_id]
        pos = origins + local
        yaw = cube_yaw[scenario_idx, cube_id]
        _write_cube_pose(env, env_ids, cube_name, pos, yaw=yaw)

    base = stack_base_local[scenario_idx]
    pose_term: HLStackPoseCommand = env.command_manager.get_term(pose_cmd_name)
    pose_term.set_stack_base(env_ids, base)
