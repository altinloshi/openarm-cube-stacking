"""Observation helpers for the OpenArm cube stacking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CUBE_NAMES: tuple[str, ...] = tuple(f"cube_{index}" for index in range(5))
NUM_CUBES = len(CUBE_NAMES)
CUBE_SIZE = 0.045
TABLE_TOP_Z = 0.40
STACK_BASE_DEFAULT = (0.56, 0.0, TABLE_TOP_Z + CUBE_SIZE * 0.5)
STACK_STATE_KEY = "_openarm_cube_stack_state"
PLACEMENT_XY_THRESHOLD = 0.025
PLACEMENT_Z_THRESHOLD = 0.015


def _make_default_stack_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.tensor(STACK_BASE_DEFAULT, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)


def _ensure_task_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor | tuple[str, ...] | float]:
    """Create the per-environment task state lazily on first use."""

    state = getattr(env, STACK_STATE_KEY, None)
    needs_init = state is None
    if not needs_init:
        stack_base = state["stack_base_pos"]
        needs_init = stack_base.shape[0] != env.num_envs or stack_base.device != env.device

    if needs_init:
        state = {
            "cube_names": CUBE_NAMES,
            "cube_size": CUBE_SIZE,
            "table_top_z": TABLE_TOP_Z,
            "stack_base_pos": _make_default_stack_base(env),
            "best_stacked_count": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        }
        setattr(env, STACK_STATE_KEY, state)

    return state


def _cube_positions_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = _ensure_task_state(env)
    cube_names = state["cube_names"]
    return torch.stack([env.scene[cube_name].data.root_pos_w[:, :3] for cube_name in cube_names], dim=1)


def _cube_orientations_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = _ensure_task_state(env)
    cube_names = state["cube_names"]
    return torch.stack([env.scene[cube_name].data.root_quat_w[:, :4] for cube_name in cube_names], dim=1)


def stack_base_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = _ensure_task_state(env)
    return state["stack_base_pos"]


def cube_target_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    base = stack_base_position(env).unsqueeze(1)
    offsets = torch.zeros((env.num_envs, NUM_CUBES, 3), device=env.device, dtype=torch.float32)
    offsets[:, :, 2] = torch.arange(NUM_CUBES, device=env.device, dtype=torch.float32) * CUBE_SIZE
    return base + offsets


def placed_cubes_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube_pos = _cube_positions_tensor(env)
    target_pos = cube_target_positions(env)
    xy_distance = torch.linalg.norm(cube_pos[:, :, :2] - target_pos[:, :, :2], dim=-1)
    z_distance = torch.abs(cube_pos[:, :, 2] - target_pos[:, :, 2])
    return (xy_distance < PLACEMENT_XY_THRESHOLD) & (z_distance < PLACEMENT_Z_THRESHOLD)


def placed_prefix_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return placed_cubes_mask(env).to(torch.int64).cumprod(dim=1).to(torch.bool)


def sequential_placed_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    return placed_prefix_mask(env).sum(dim=1)


def update_best_stacked_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = _ensure_task_state(env)
    current_count = sequential_placed_count(env)
    state["best_stacked_count"] = torch.maximum(state["best_stacked_count"], current_count)
    return state["best_stacked_count"]


def current_cube_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    return sequential_placed_count(env).unsqueeze(-1).to(torch.float32)


def _active_cube_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(sequential_placed_count(env), max=NUM_CUBES - 1)


def cube_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _cube_positions_tensor(env).reshape(env.num_envs, -1)


def cube_orientations(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _cube_orientations_tensor(env).reshape(env.num_envs, -1)


def ee_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    return ee_frame.data.target_pos_w[:, 0, :]


def ee_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    return torch.cat((ee_pos, ee_quat), dim=-1)


def current_cube_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube_pos = _cube_positions_tensor(env)
    active_idx = _active_cube_index(env)
    return cube_pos[torch.arange(env.num_envs, device=env.device), active_idx]


def current_cube_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube_quat = _cube_orientations_tensor(env)
    active_idx = _active_cube_index(env)
    return cube_quat[torch.arange(env.num_envs, device=env.device), active_idx]


def current_target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    target_pos = cube_target_positions(env)
    active_idx = _active_cube_index(env)
    return target_pos[torch.arange(env.num_envs, device=env.device), active_idx]


def ee_to_current_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    return current_cube_position(env) - ee_position(env)


def current_cube_to_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    return current_target_position(env) - current_cube_position(env)
