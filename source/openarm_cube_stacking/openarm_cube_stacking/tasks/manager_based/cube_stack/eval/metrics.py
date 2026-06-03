from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

from ..tabletop_scene import CUBE_NAMES, CUBE_SIZE, NUM_CUBES, TABLE_TOP_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class StackMetrics:
    task: str
    num_envs: int
    episodes: int
    full_stack_success_rate: float = 0.0
    average_cubes_stacked: float = 0.0
    per_cube_success_rate: list[float] | None = None
    cube_drop_rate: float = 0.0
    stack_collapse_rate: float = 0.0
    mean_final_stack_error: float = 0.0
    mean_episode_length_s: float = 0.0
    mean_grasp_retries: float = 0.0
    timeout_rate: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        if data["per_cube_success_rate"] is None:
            data["per_cube_success_rate"] = [0.0] * NUM_CUBES
        return data


def _cube_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.stack([env.scene[name].data.root_pos_w[:, :3] for name in CUBE_NAMES], dim=1)


def _cube_velocities(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.stack([env.scene[name].data.root_lin_vel_w[:, :3] for name in CUBE_NAMES], dim=1)


def target_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "stack_base_pos_w"):
        base = env.stack_base_pos_w
    else:
        pose_term = env.command_manager.get_term("ee_pose")
        base = pose_term.stack_base_pos_w
    targets = base[:, None, :].repeat(1, NUM_CUBES, 1)
    z_offsets = torch.arange(NUM_CUBES, device=env.device, dtype=torch.float32) * CUBE_SIZE
    targets[:, :, 2] += z_offsets[None, :]
    return targets


def compute_stack_success_mask(
    env: "ManagerBasedRLEnv",
    position_tolerance: float = 0.035,
    height_tolerance: float = 0.02,
    velocity_threshold: float = 0.03,
) -> torch.Tensor:
    """Per-cube stable stack success mask, shape ``(num_envs, 5)``."""
    cube_pos = _cube_positions(env)
    cube_vel = _cube_velocities(env)
    targets = target_positions(env)
    xy_ok = torch.norm(cube_pos[:, :, :2] - targets[:, :, :2], dim=-1) < position_tolerance
    z_ok = (cube_pos[:, :, 2] - targets[:, :, 2]).abs() < height_tolerance
    stable = torch.norm(cube_vel, dim=-1) < velocity_threshold
    placed = xy_ok & z_ok & stable
    # A cube only counts if every cube below it is also stable.
    return torch.cumprod(placed.to(torch.int32), dim=1).to(torch.bool)


def final_stack_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.norm(_cube_positions(env) - target_positions(env), dim=-1).mean(dim=1)


def cube_drop_mask(env: "ManagerBasedRLEnv", min_height: float = TABLE_TOP_Z - 0.08, workspace_radius: float = 0.90) -> torch.Tensor:
    cube_pos = _cube_positions(env)
    origins = env.scene.env_origins.to(env.device)[:, None, :]
    local_xy = cube_pos[:, :, :2] - origins[:, :, :2]
    return ((cube_pos[:, :, 2] < min_height) | (torch.norm(local_xy, dim=-1) > workspace_radius)).any(dim=1)


def stack_collapse_mask(env: "ManagerBasedRLEnv", threshold: float = 0.065) -> torch.Tensor:
    placed = compute_stack_success_mask(env, position_tolerance=threshold)
    # A collapsed stack has at least one lower cube missing while a higher cube is also not successful.
    any_started = placed.any(dim=1)
    full = placed.all(dim=1)
    return any_started & ~full


def summarize_episode_batch(
    env: "ManagerBasedRLEnv",
    task: str,
    episodes: int,
    timeout_mask: torch.Tensor | None = None,
) -> StackMetrics:
    placed = compute_stack_success_mask(env)
    full = placed.all(dim=1)
    pose_term = env.command_manager.get_term("ee_pose") if hasattr(env, "command_manager") else None
    retries = pose_term.planner.total_retries.float() if pose_term is not None else torch.zeros(env.num_envs, device=env.device)
    timeout = timeout_mask.float().mean().item() if timeout_mask is not None else 0.0
    return StackMetrics(
        task=task,
        num_envs=env.num_envs,
        episodes=episodes,
        full_stack_success_rate=full.float().mean().item(),
        average_cubes_stacked=placed.float().sum(dim=1).mean().item(),
        per_cube_success_rate=placed.float().mean(dim=0).detach().cpu().tolist(),
        cube_drop_rate=cube_drop_mask(env).float().mean().item(),
        stack_collapse_rate=stack_collapse_mask(env).float().mean().item(),
        mean_final_stack_error=final_stack_error(env).mean().item(),
        mean_episode_length_s=float(getattr(env, "episode_length_s", 0.0)),
        mean_grasp_retries=retries.mean().item(),
        timeout_rate=timeout,
    )
