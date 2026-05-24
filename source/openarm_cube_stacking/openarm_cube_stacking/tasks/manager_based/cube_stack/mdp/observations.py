"""Observation helpers for the OpenArm 5-cube stacking task.

All helpers operate on vectorized batches ``(num_envs, ...)`` and avoid Python
loops over environments. ``env`` is expected to expose:

* ``env.scene[<cube_name>]`` – :class:`isaaclab.assets.RigidObject` for each cube
* ``env.scene["robot"]``     – the OpenArm articulation
* ``env.scene["ee_frame"]``  – :class:`isaaclab.sensors.FrameTransformer` for the TCP
* ``env.cfg.num_cubes``        – number of cubes (default 5)
* ``env.cfg.cube_size``        – per-side length of a cube in meters
* ``env.cfg.stack_base_pos``   – ``(x, y, z)`` world-frame base of the target stack
* ``env.cfg.place_threshold``  – distance below which a cube counts as ``placed``

The current-cube logic is implemented as a buffer ``env.current_cube_idx``
(``num_envs`` long, ``int64``) that is computed lazily on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cube_names(env: "ManagerBasedRLEnv") -> list[str]:
    """Return the ordered list of cube scene-entity names."""
    num_cubes = int(getattr(env.cfg, "num_cubes", 5))
    return [f"cube_{i}" for i in range(num_cubes)]


def _gather_cube_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Stacked cube positions ``(num_envs, num_cubes, 3)`` in world frame."""
    cubes = [env.scene[name] for name in _cube_names(env)]
    return torch.stack([c.data.root_pos_w[:, :3] for c in cubes], dim=1)


def _gather_cube_quats(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Stacked cube quaternions ``(num_envs, num_cubes, 4)`` (wxyz) in world frame."""
    cubes = [env.scene[name] for name in _cube_names(env)]
    return torch.stack([c.data.root_quat_w for c in cubes], dim=1)


def _ee_position(env: "ManagerBasedRLEnv", ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector world position ``(num_envs, 3)``."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def _stack_base(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env stack base position ``(num_envs, 3)``.

    Stored as a buffer so the events module can randomize it on reset.
    """
    if not hasattr(env, "stack_base_pos") or env.stack_base_pos.shape[0] != env.num_envs:
        base = torch.tensor(
            getattr(env.cfg, "stack_base_pos", (0.5, 0.0, 0.0)),
            device=env.device,
            dtype=torch.float32,
        )
        env.stack_base_pos = base.unsqueeze(0).expand(env.num_envs, -1).clone()
    return env.stack_base_pos


def _target_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target world position for every cube index ``(num_envs, num_cubes, 3)``.

    ``cube_i`` should be placed at ``stack_base + [0, 0, (i + 0.5) * cube_size]``.
    """
    num_cubes = int(getattr(env.cfg, "num_cubes", 5))
    cube_size = float(getattr(env.cfg, "cube_size", 0.05))
    base = _stack_base(env)  # (N, 3)
    idx = torch.arange(num_cubes, device=env.device, dtype=torch.float32)
    z_offsets = (idx + 0.5) * cube_size  # (num_cubes,)
    targets = base.unsqueeze(1).expand(-1, num_cubes, -1).clone()
    targets[..., 2] = base[:, 2:3] + z_offsets.unsqueeze(0)
    return targets


def _current_cube_idx(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Index of the cube the policy should currently move ``(num_envs,)`` long.

    The "current cube" is the lowest-index cube that is **not yet placed**.
    A cube is considered placed when its xy distance to its target is below
    ``cfg.place_threshold`` *and* its height is within ``cfg.place_threshold``
    of the target z.
    """
    num_cubes = int(getattr(env.cfg, "num_cubes", 5))
    place_thresh = float(getattr(env.cfg, "place_threshold", 0.03))

    cube_pos = _gather_cube_positions(env)        # (N, K, 3)
    targets = _target_positions(env)              # (N, K, 3)
    xy_dist = torch.norm(cube_pos[..., :2] - targets[..., :2], dim=-1)  # (N, K)
    z_dist = torch.abs(cube_pos[..., 2] - targets[..., 2])              # (N, K)
    placed = (xy_dist < place_thresh) & (z_dist < place_thresh)         # (N, K)

    # First False along K dimension == current cube. If all placed, clamp to last.
    not_placed = ~placed
    # argmax of bool returns first True; if all False (everything placed) -> 0
    idx = torch.argmax(not_placed.to(torch.int64), dim=1)
    all_placed = placed.all(dim=1)
    idx = torch.where(all_placed, torch.full_like(idx, num_cubes - 1), idx)
    env.current_cube_idx = idx
    return idx


def _gather_per_env(tensor_NK3: torch.Tensor, idx_N: torch.Tensor) -> torch.Tensor:
    """Index a ``(N, K, 3)`` tensor with a per-env index ``(N,)`` -> ``(N, 3)``."""
    return tensor_NK3.gather(1, idx_N.view(-1, 1, 1).expand(-1, 1, tensor_NK3.shape[-1])).squeeze(1)


# ---------------------------------------------------------------------------
# Public observation functions
# ---------------------------------------------------------------------------


def cube_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """All cube positions, flattened to ``(num_envs, num_cubes * 3)``."""
    return _gather_cube_positions(env).reshape(env.num_envs, -1)


def cube_orientations(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """All cube quaternions, flattened to ``(num_envs, num_cubes * 4)``."""
    return _gather_cube_quats(env).reshape(env.num_envs, -1)


def current_cube_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Position of the currently-active cube ``(num_envs, 3)``."""
    cube_pos = _gather_cube_positions(env)
    idx = _current_cube_idx(env)
    return _gather_per_env(cube_pos, idx)


def current_target_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target position for the currently-active cube ``(num_envs, 3)``."""
    targets = _target_positions(env)
    idx = _current_cube_idx(env)
    return _gather_per_env(targets, idx)


def ee_to_current_cube(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from the end-effector to the currently-active cube ``(num_envs, 3)``."""
    ee = _ee_position(env, ee_frame_cfg)
    return current_cube_position(env) - ee


def current_cube_to_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vector from the current cube to its target position ``(num_envs, 3)``."""
    return current_target_position(env) - current_cube_position(env)


def current_cube_index(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Index of the currently-active cube as a float column ``(num_envs, 1)``."""
    return _current_cube_idx(env).to(torch.float32).unsqueeze(-1)


def ee_position(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector world position ``(num_envs, 3)``."""
    return _ee_position(env, ee_frame_cfg)


def stack_base_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """The current per-env target stack base position ``(num_envs, 3)``."""
    return _stack_base(env).clone()
