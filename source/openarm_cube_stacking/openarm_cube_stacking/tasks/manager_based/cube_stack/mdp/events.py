"""Event/reset helpers for the OpenArm cube-stacking task.

These helpers follow the Isaac Lab event-term signature ``func(env, env_ids, **params)``.
For the first version, cube spawn positions are deterministic (laid out on a grid
beside the stack base) but the helpers are structured so randomization can be
added simply by enabling the ``randomize`` parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: "ManagerBasedRLEnv", env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def reset_robot_to_default(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot joints (and root pose) to the articulation's default state."""
    ids = _resolve_env_ids(env, env_ids)
    robot: Articulation = env.scene[asset_cfg.name]

    joint_pos = robot.data.default_joint_pos[ids].clone()
    joint_vel = robot.data.default_joint_vel[ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    root_state = robot.data.default_root_state[ids].clone()
    root_state[:, :3] += env.scene.env_origins[ids]
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)


def reset_stack_target(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | None,
    base_pos: tuple[float, float, float] | None = None,
    pos_range_xy: tuple[float, float] = (0.0, 0.0),
) -> None:
    """(Re)initialize the per-env stack-target base position buffer.

    Use ``pos_range_xy = (-r, r)`` to add uniform noise around ``base_pos``.
    """
    ids = _resolve_env_ids(env, env_ids)
    base = torch.tensor(
        base_pos if base_pos is not None else getattr(env.cfg, "stack_base_pos", (0.5, 0.0, 0.0)),
        device=env.device,
        dtype=torch.float32,
    )
    if not hasattr(env, "stack_base_pos") or env.stack_base_pos.shape[0] != env.num_envs:
        env.stack_base_pos = base.unsqueeze(0).expand(env.num_envs, -1).clone()

    new_base = base.unsqueeze(0).expand(ids.numel(), -1).clone()
    if pos_range_xy[1] > pos_range_xy[0]:
        lo, hi = pos_range_xy
        new_base[:, 0] += torch.empty(ids.numel(), device=env.device).uniform_(lo, hi)
        new_base[:, 1] += torch.empty(ids.numel(), device=env.device).uniform_(lo, hi)
    env.stack_base_pos[ids] = new_base


def _grid_offsets(num_cubes: int, cube_size: float, gap: float = 0.02) -> torch.Tensor:
    """Deterministic non-overlapping XY offsets for the cubes (a row beside the stack).

    Cube ``i`` is placed at ``y = (i - (K-1)/2) * (cube_size + gap)``, ``x = -2 * cube_size``.
    """
    pitch = cube_size + gap
    idx = torch.arange(num_cubes, dtype=torch.float32)
    offsets = torch.zeros(num_cubes, 3)
    offsets[:, 0] = -2.0 * cube_size
    offsets[:, 1] = (idx - (num_cubes - 1) / 2.0) * pitch
    offsets[:, 2] = 0.5 * cube_size  # rest on the table surface
    return offsets


def reset_cubes_non_overlapping(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | None,
    randomize: bool = False,
    pos_noise: float = 0.0,
) -> None:
    """Reset every cube to a deterministic, non-overlapping spawn pose.

    The cubes are laid out in a row next to the stack base.  When ``randomize``
    is ``True``, uniform noise of magnitude ``pos_noise`` is added in xy.
    """
    ids = _resolve_env_ids(env, env_ids)
    num_cubes = int(getattr(env.cfg, "num_cubes", 5))
    cube_size = float(getattr(env.cfg, "cube_size", 0.05))

    base = torch.tensor(
        getattr(env.cfg, "stack_base_pos", (0.5, 0.0, 0.0)),
        device=env.device,
        dtype=torch.float32,
    )
    offsets = _grid_offsets(num_cubes, cube_size).to(env.device)

    for i in range(num_cubes):
        cube: RigidObject = env.scene[f"cube_{i}"]
        # default state: (root_state_w stored in default_root_state)
        root_state = cube.data.default_root_state[ids].clone()
        root_state[:, :3] = env.scene.env_origins[ids] + base + offsets[i]
        if randomize and pos_noise > 0.0:
            noise = torch.empty(ids.numel(), 2, device=env.device).uniform_(-pos_noise, pos_noise)
            root_state[:, :2] += noise
        # zero velocities
        root_state[:, 7:] = 0.0
        cube.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
        cube.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)
