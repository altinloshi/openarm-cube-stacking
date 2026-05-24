"""Event (reset) functions for the 5-cube stacking task.

Event functions are called by the EventManager during resets.  The signature
differs from observation/reward terms:

    func(env: ManagerBasedRLEnv, env_ids: torch.Tensor, **params) -> None

``env_ids`` contains the indices of environments being reset.

Most standard resets (joint positions, root states with uniform noise) are
already provided by ``isaaclab.envs.mdp`` and are wired up in EventCfg via
``reset_scene_to_default`` and ``reset_root_state_uniform``.  The functions
below implement task-specific logic that goes beyond what the standard events
offer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import CUBE_NAMES, CUBE_SIZE, NUM_CUBES, STACK_BASE_LOCAL_Z


# ---------------------------------------------------------------------------
# Custom event helpers
# ---------------------------------------------------------------------------


def reset_cubes_non_overlapping(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.25, 0.45),
    y_range: tuple[float, float] = (-0.30, 0.30),
    min_separation: float = 0.12,
    max_attempts: int = 200,
) -> None:
    """Reset all 5 cubes to random, non-overlapping positions on the table.

    Each cube's centre is sampled uniformly from [x_range] × [y_range] in
    env-local coordinates and placed at the table-surface height.  Positions
    are rejected and re-sampled if any two cubes are closer than
    ``min_separation`` in the XY plane.

    Args:
        env: The RL environment.
        env_ids: Indices of environments to reset.
        x_range: (min, max) x offset from env origin.
        y_range: (min, max) y offset from env origin.
        min_separation: Minimum allowed XY distance between cube centres (m).
        max_attempts: Maximum rejection-sampling iterations before giving up.
    """
    n = len(env_ids)
    device = env.device
    table_z = STACK_BASE_LOCAL_Z  # same surface height used for stack base

    for cube_idx, cube_name in enumerate(CUBE_NAMES):
        cube: "RigidObject" = env.scene[cube_name]  # type: ignore[name-defined]

        # Sample candidate positions for this cube across all reset envs
        success = torch.zeros(n, dtype=torch.bool, device=device)
        new_xy = torch.zeros(n, 2, device=device)

        for _ in range(max_attempts):
            # Sample candidates for envs not yet placed
            remaining = ~success
            count = remaining.sum().item()
            if count == 0:
                break

            cand_x = torch.rand(count, device=device) * (x_range[1] - x_range[0]) + x_range[0]
            cand_y = torch.rand(count, device=device) * (y_range[1] - y_range[0]) + y_range[0]
            cand_xy = torch.stack([cand_x, cand_y], dim=-1)  # [count, 2]

            # Check separation from already-placed cubes (indices < cube_idx)
            valid = torch.ones(count, dtype=torch.bool, device=device)
            for prev_idx in range(cube_idx):
                prev_cube = env.scene[CUBE_NAMES[prev_idx]]
                prev_pos_w = prev_cube.data.root_pos_w[env_ids[remaining]]  # [count, 3]
                prev_xy_local = prev_pos_w[:, :2] - env.scene.env_origins[env_ids[remaining], :2]
                diff = cand_xy - prev_xy_local  # [count, 2]
                dist = torch.norm(diff, dim=-1)  # [count]
                valid &= dist >= min_separation

            # Also check separation from other stacks/targets
            # (avoid spawning at the stack target)
            stack_x = 0.55
            stack_y = 0.0
            dist_to_stack = torch.norm(cand_xy - torch.tensor([stack_x, stack_y], device=device), dim=-1)
            valid &= dist_to_stack >= min_separation

            rem_indices = remaining.nonzero(as_tuple=False).squeeze(-1)
            accepted = valid
            new_xy[rem_indices[accepted]] = cand_xy[accepted]
            success[rem_indices[accepted]] = True

        # Apply positions
        new_pos_w = env.scene.env_origins[env_ids].clone()  # [n, 3]
        new_pos_w[:, 0] += new_xy[:, 0]
        new_pos_w[:, 1] += new_xy[:, 1]
        new_pos_w[:, 2] += table_z

        # Write root state
        root_state = cube.data.default_root_state[env_ids].clone()
        root_state[:, :3] = new_pos_w
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # identity quat
        root_state[:, 7:] = 0.0  # zero velocity

        cube.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        cube.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_stack_target(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    randomise: bool = False,
    x_range: tuple[float, float] = (0.50, 0.60),
    y_range: tuple[float, float] = (-0.05, 0.05),
) -> None:
    """(Optional) Store a per-env stack base position on the environment object.

    When ``randomise=True``, each env gets a slightly different stack target
    so the policy learns to generalise across target locations.  The position
    is stored as ``env._stack_base_local`` for use by observation/reward
    functions.

    NOTE: The current observations/rewards use a fixed stack base and do NOT
    read ``env._stack_base_local``.  This function is a scaffold for future
    randomised-target curriculum.  To enable it, wire it into EventCfg and
    update the helper ``_stack_target_w`` in observations.py to read this
    attribute.
    """
    if not randomise:
        return  # Nothing to do; fixed base is hardcoded in observations.py

    n = len(env_ids)
    base_x = torch.rand(n, device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
    base_y = torch.rand(n, device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
    base_z = torch.full((n,), STACK_BASE_LOCAL_Z, device=env.device)
    base_local = torch.stack([base_x, base_y, base_z], dim=-1)  # [n, 3]

    if not hasattr(env, "_stack_base_local"):
        env._stack_base_local = torch.zeros(env.num_envs, 3, device=env.device)
    env._stack_base_local[env_ids] = base_local


def reset_robot_to_default(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot joint positions and velocities to their default values.

    This is a thin wrapper; in most cases ``reset_scene_to_default`` already
    handles this.  Provided here as a standalone hook if needed.
    """
    robot = env.scene[robot_cfg.name]
    default_state = robot.data.default_joint_pos[env_ids]
    robot.write_joint_state_to_sim(default_state, torch.zeros_like(default_state), env_ids=env_ids)
