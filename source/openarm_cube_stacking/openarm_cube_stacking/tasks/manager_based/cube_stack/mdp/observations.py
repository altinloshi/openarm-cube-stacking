from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NUM_CUBES = 5
CUBE_SIZE = 0.05
TABLE_HEIGHT = 0.40
TABLE_TOP_Z = TABLE_HEIGHT
CUBE_CENTER_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0
CUBE_NAMES = tuple(f"cube_{i}" for i in range(NUM_CUBES))

DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS = (
    (0.34, -0.18, CUBE_CENTER_Z),
    (0.34, -0.09, CUBE_CENTER_Z),
    (0.34, 0.00, CUBE_CENTER_Z),
    (0.34, 0.09, CUBE_CENTER_Z),
    (0.34, 0.18, CUBE_CENTER_Z),
)
DEFAULT_STACK_BASE_LOCAL_POS = (0.55, 0.0, CUBE_CENTER_Z)


def _scene_origins(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return cloned environment origins on the task device."""
    return env.scene.env_origins.to(env.device)


def _cube_positions_w(env: ManagerBasedRLEnv, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Stack cube root positions as ``(num_envs, num_cubes, 3)``."""
    positions = [env.scene[name].data.root_pos_w[:, :3] for name in cube_names]
    return torch.stack(positions, dim=1)


def _cube_orientations_w(env: ManagerBasedRLEnv, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Stack cube root orientations as ``(num_envs, num_cubes, 4)``."""
    orientations = [env.scene[name].data.root_quat_w[:, :4] for name in cube_names]
    return torch.stack(orientations, dim=1)


def _stack_base_position_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-env stack base position in world frame.

    The reset event owns this state. Observation/reward helpers create a deterministic default so the
    task remains well-defined before the first reset event is executed.
    """
    if not hasattr(env, "stack_base_pos_w") or env.stack_base_pos_w.shape[0] != env.num_envs:
        local_pos = torch.tensor(DEFAULT_STACK_BASE_LOCAL_POS, device=env.device).repeat(env.num_envs, 1)
        env.stack_base_pos_w = _scene_origins(env) + local_pos
    return env.stack_base_pos_w


def _target_positions_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return target cube-center positions as ``(num_envs, num_cubes, 3)``."""
    stack_base = _stack_base_position_w(env)
    z_offsets = torch.arange(NUM_CUBES, device=env.device, dtype=torch.float32) * CUBE_SIZE
    targets = stack_base[:, None, :].repeat(1, NUM_CUBES, 1)
    targets[:, :, 2] += z_offsets[None, :]
    return targets


def _placed_mask(env: ManagerBasedRLEnv, threshold: float = 0.035) -> torch.Tensor:
    """Return a boolean mask for cubes that are close to their sequential stack targets."""
    cube_pos = _cube_positions_w(env)
    target_pos = _target_positions_w(env)
    return torch.norm(cube_pos - target_pos, dim=-1) < threshold


def _current_cube_indices(env: ManagerBasedRLEnv, threshold: float = 0.035) -> torch.Tensor:
    """Return the first cube index that has not reached its stack target."""
    placed = _placed_mask(env, threshold=threshold)
    not_placed = ~placed
    current_idx = torch.argmax(not_placed.to(torch.int64), dim=1)
    all_placed = placed.all(dim=1)
    return torch.where(all_placed, torch.full_like(current_idx, NUM_CUBES - 1), current_idx)


def _gather_current(values: torch.Tensor, current_idx: torch.Tensor) -> torch.Tensor:
    """Gather per-cube values at the current cube index."""
    gather_shape = [values.shape[0], 1, *values.shape[2:]]
    index = current_idx.reshape(values.shape[0], 1, *([1] * (values.ndim - 2))).expand(*gather_shape)
    return torch.gather(values, dim=1, index=index).squeeze(1)


def cube_positions(env: ManagerBasedRLEnv, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Cube positions in world frame, flattened as ``(num_envs, 15)``."""
    return _cube_positions_w(env, cube_names=cube_names).reshape(env.num_envs, -1)


def cube_orientations(env: ManagerBasedRLEnv, cube_names: Sequence[str] = CUBE_NAMES) -> torch.Tensor:
    """Cube orientations in world frame, flattened as ``(num_envs, 20)``."""
    return _cube_orientations_w(env, cube_names=cube_names).reshape(env.num_envs, -1)


def end_effector_position(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector target frame position in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def end_effector_orientation(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector target frame orientation in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]


def current_cube_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current sequential cube index as a scalar observation."""
    return _current_cube_indices(env).to(torch.float32).unsqueeze(-1)


def current_cube_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Position of the first cube that is not yet stacked."""
    cube_pos = _cube_positions_w(env)
    current_idx = _current_cube_indices(env)
    return _gather_current(cube_pos, current_idx)


def current_target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target position for the current cube."""
    target_pos = _target_positions_w(env)
    current_idx = _current_cube_indices(env)
    return _gather_current(target_pos, current_idx)


def ee_to_current_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the end-effector to the current cube."""
    return current_cube_position(env) - end_effector_position(env)


def current_cube_to_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the current cube to its current stack target."""
    return current_target_position(env) - current_cube_position(env)


def stack_target_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """All target stack positions flattened as ``(num_envs, 15)``."""
    return _target_positions_w(env).reshape(env.num_envs, -1)


def stack_base_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-environment stack base position in world frame."""
    return _stack_base_position_w(env)

