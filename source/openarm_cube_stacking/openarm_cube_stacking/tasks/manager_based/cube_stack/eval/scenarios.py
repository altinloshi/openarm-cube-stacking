"""30 deterministic evaluation scenarios for tournament scoring.

Each scenario fixes:
- Five cube initial positions on the table (local env frame)
- Stack base XY target position (local env frame)
- Optional per-cube yaw offset

Scenarios are indexed 0–29.  Environment i uses scenario ``i % 30``.

Layout conventions
------------------
All positions are in the LOCAL env frame (env origins are added at runtime).
Cubes rest on the table: z = TABLE_TOP_Z + CUBE_SIZE / 2.0 = 0.225 m.
Stack base z = 0.225 m (bottom cube centre on table top).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..tabletop_scene_cfg import CUBE_SIZE, CUBE_TABLE_Z, CUBE_NAMES, TABLE_TOP_Z

# ─────────────────────────────────────────────────────────────────────────────
# Scenario definition type
# ─────────────────────────────────────────────────────────────────────────────

# Each scenario: dict with keys
#   "cubes":  list of 5 (x, y, z) positions in local frame
#   "stack":  (x, y, z) stack base position in local frame
#   "yaws":   list of 5 yaw angles in radians (optional, 0 = upright)

_Z = CUBE_TABLE_Z  # 0.225


def _mk_scenario(
    cube_xs: Sequence[float],
    cube_ys: Sequence[float],
    stack_x: float,
    stack_y: float,
    yaws: Sequence[float] | None = None,
) -> dict:
    return {
        "cubes": [(x, y, _Z) for x, y in zip(cube_xs, cube_ys)],
        "stack": (stack_x, stack_y, _Z),
        "yaws": yaws if yaws is not None else [0.0] * 5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 30 deterministic scenarios
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SCENARIOS: list[dict] = [
    # ── Baseline row layout, various stack positions ──────────────────────────
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, 0.0),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, 0.10),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, -0.10),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.80, 0.15),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.80, -0.15),

    # ── Cubes spread wider in y ───────────────────────────────────────────────
    _mk_scenario([0.73]*5, [-0.20, -0.10, 0.0, 0.10, 0.20], 0.87, 0.0),
    _mk_scenario([0.73]*5, [-0.20, -0.10, 0.0, 0.10, 0.20], 0.87, 0.10),
    _mk_scenario([0.73]*5, [-0.20, -0.10, 0.0, 0.10, 0.20], 0.87, -0.10),

    # ── Cubes in a 2-column arrangement ──────────────────────────────────────
    _mk_scenario([0.70, 0.70, 0.77, 0.77, 0.73],
                 [-0.12, 0.12, -0.12, 0.12, 0.0], 0.85, 0.0),
    _mk_scenario([0.70, 0.70, 0.77, 0.77, 0.73],
                 [-0.12, 0.12, -0.12, 0.12, 0.0], 0.85, -0.12),

    # ── Varied x-positions (cubes not on same row) ───────────────────────────
    _mk_scenario([0.70, 0.73, 0.76, 0.79, 0.75],
                 [-0.15, -0.07, 0.0, 0.07, 0.15], 0.85, 0.0),
    _mk_scenario([0.71, 0.74, 0.77, 0.80, 0.74],
                 [-0.14, -0.07, 0.0, 0.07, 0.14], 0.83, 0.05),

    # ── Stack target moved to corners of reachable zone ───────────────────────
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.88, 0.18),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.88, -0.18),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.78, 0.0),

    # ── Cubes with yaw variation ──────────────────────────────────────────────
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, 0.0,
                 yaws=[0.0, math.pi/4, -math.pi/4, math.pi/6, -math.pi/6]),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, 0.08,
                 yaws=[math.pi/8]*5),
    _mk_scenario([0.75]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, -0.08,
                 yaws=[-math.pi/8, math.pi/8, 0.0, math.pi/4, -math.pi/4]),

    # ── Compact cube cluster ──────────────────────────────────────────────────
    _mk_scenario([0.74, 0.75, 0.76, 0.74, 0.76],
                 [-0.08, -0.04, 0.0, 0.04, 0.08], 0.85, 0.0),
    _mk_scenario([0.73, 0.74, 0.75, 0.74, 0.73],
                 [-0.08, -0.04, 0.0, 0.04, 0.08], 0.86, 0.0),

    # ── Asymmetric cube arrangements ──────────────────────────────────────────
    _mk_scenario([0.70, 0.72, 0.74, 0.76, 0.78],
                 [0.0, 0.0, 0.0, 0.0, 0.0], 0.85, 0.0),
    _mk_scenario([0.75, 0.75, 0.75, 0.75, 0.75],
                 [-0.15, 0.15, -0.10, 0.10, 0.0], 0.87, 0.0),
    _mk_scenario([0.72, 0.78, 0.72, 0.78, 0.75],
                 [-0.12, -0.12, 0.12, 0.12, 0.0], 0.85, 0.0),
    _mk_scenario([0.73, 0.73, 0.73, 0.73, 0.73],
                 [-0.16, -0.08, 0.0, 0.08, 0.16], 0.85, -0.05),

    # ── Challenging reach: cubes further forward ──────────────────────────────
    _mk_scenario([0.78]*5, [-0.15, -0.07, 0.0, 0.07, 0.15], 0.90, 0.0),
    _mk_scenario([0.76]*5, [-0.15, -0.07, 0.0, 0.07, 0.15], 0.88, 0.10),

    # ── Mixed x/y with varied stack ──────────────────────────────────────────
    _mk_scenario([0.71, 0.74, 0.77, 0.74, 0.71],
                 [-0.14, -0.07, 0.0, 0.07, 0.14], 0.84, 0.0),
    _mk_scenario([0.73, 0.76, 0.73, 0.76, 0.74],
                 [-0.12, -0.06, 0.0, 0.06, 0.12], 0.86, 0.06),
    _mk_scenario([0.74, 0.74, 0.74, 0.74, 0.74],
                 [-0.18, -0.09, 0.0, 0.09, 0.18], 0.85, -0.06),

    # ── Scenario 29: maximum separation ──────────────────────────────────────
    _mk_scenario([0.70, 0.72, 0.74, 0.76, 0.78],
                 [-0.18, -0.09, 0.0, 0.09, 0.18], 0.88, 0.0),
]

assert len(EVAL_SCENARIOS) == 30, f"Expected 30 scenarios, got {len(EVAL_SCENARIOS)}"


# ─────────────────────────────────────────────────────────────────────────────
# Reset functions for use as Isaac Lab EventTerm funcs
# ─────────────────────────────────────────────────────────────────────────────


def _yaw_to_quat(yaw: float, device: torch.device) -> torch.Tensor:
    """Convert yaw angle to quaternion (wxyz), rotation around Z axis."""
    half = yaw / 2.0
    return torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=torch.float32,
        device=device,
    )


def reset_cubes_from_scenario(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    cube_names: Sequence[str] = list(CUBE_NAMES),
    scenarios: list[dict] = EVAL_SCENARIOS,
) -> None:
    """Reset cubes to the deterministic positions for each env's scenario."""
    from isaaclab.assets import RigidObject

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    origins = env.scene.env_origins[env_ids]
    velocity = torch.zeros((len(env_ids), 6), device=env.device)

    for local_i, env_idx in enumerate(env_ids.tolist()):
        scenario = scenarios[int(env_idx) % len(scenarios)]
        cube_positions = scenario["cubes"]
        yaws = scenario["yaws"]

        for cube_id, cube_name in enumerate(cube_names):
            cube: RigidObject = env.scene[cube_name]
            pos = origins[local_i] + torch.tensor(
                cube_positions[cube_id], dtype=torch.float32, device=env.device
            )
            quat = _yaw_to_quat(yaws[cube_id], env.device)
            pose = torch.cat([pos.unsqueeze(0), quat.unsqueeze(0)], dim=-1)
            single_id = env_ids[local_i:local_i+1]
            cube.write_root_pose_to_sim(pose, env_ids=single_id)
            cube.write_root_velocity_to_sim(velocity[local_i:local_i+1], env_ids=single_id)
            cube.reset(single_id)


def reset_stack_from_scenario(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    scenarios: list[dict] = EVAL_SCENARIOS,
) -> None:
    """Reset the stack base position from the scenario lookup table."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    if not hasattr(env, "stack_base_pos_w") or env.stack_base_pos_w.shape[0] != env.num_envs:
        env.stack_base_pos_w = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)

    origins = env.scene.env_origins[env_ids]

    for local_i, env_idx in enumerate(env_ids.tolist()):
        scenario = scenarios[int(env_idx) % len(scenarios)]
        stack_local = torch.tensor(scenario["stack"], dtype=torch.float32, device=env.device)
        env.stack_base_pos_w[env_idx] = origins[local_i] + stack_local
