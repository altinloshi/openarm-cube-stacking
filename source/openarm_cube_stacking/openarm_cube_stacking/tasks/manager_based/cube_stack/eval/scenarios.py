"""30 deterministic evaluation scenarios for tournament scoring.

Each scenario fixes:
- Five cube initial positions in the local env frame
- Stack base XY target position in the local env frame
- Optional per-cube yaw offset

Scenarios are indexed 0–29.  Environment i uses scenario ``i % 30``.

Layout conventions
------------------
All positions are in the LOCAL env frame (env origins are added at runtime).
OpenArm is floor-mounted at env origin (0, 0, 0) – no table.
Cubes rest on the floor: z = CUBE_GROUND_Z = 0.055 m (centre).
Stack base z = 0.055 m (centre of the first cube in the stack).

Workspace limits (floor-mounted OpenArm)
-----------------------------------------
  cube x : 0.28 – 0.45 m   (forward reach from robot base)
  cube y : -0.20 – 0.20 m  (lateral reach)
  stack x: 0.44 – 0.56 m   (slightly further than cubes)
  stack y: -0.14 – 0.14 m
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..openarm_lift_style_scene_cfg import CUBE_GROUND_Z, CUBE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Scenario definition
# ─────────────────────────────────────────────────────────────────────────────

_Z = CUBE_GROUND_Z  # 0.055 m – cube centre when resting on floor


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
# 30 deterministic scenarios  (floor-mounted OpenArm workspace)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SCENARIOS: list[dict] = [
    # ── Standard row, various stack positions ─────────────────────────────────
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.52, 0.0),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.52, 0.08),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.52, -0.08),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.48, 0.12),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.48, -0.12),

    # ── Wider y spread ────────────────────────────────────────────────────────
    _mk_scenario([0.33]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.52, 0.0),
    _mk_scenario([0.33]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.52, 0.08),
    _mk_scenario([0.33]*5, [-0.18, -0.09, 0.0, 0.09, 0.18], 0.52, -0.08),

    # ── 2-column arrangement ──────────────────────────────────────────────────
    _mk_scenario([0.30, 0.30, 0.37, 0.37, 0.33],
                 [-0.10, 0.10, -0.10, 0.10, 0.0], 0.50, 0.0),
    _mk_scenario([0.30, 0.30, 0.37, 0.37, 0.33],
                 [-0.10, 0.10, -0.10, 0.10, 0.0], 0.50, -0.08),

    # ── Varied x positions ────────────────────────────────────────────────────
    _mk_scenario([0.30, 0.33, 0.36, 0.39, 0.35],
                 [-0.14, -0.07, 0.0, 0.07, 0.14], 0.52, 0.0),
    _mk_scenario([0.31, 0.34, 0.37, 0.40, 0.34],
                 [-0.13, -0.06, 0.0, 0.06, 0.13], 0.50, 0.04),

    # ── Stack target toward workspace edges ───────────────────────────────────
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.53, 0.14),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.53, -0.14),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.46, 0.0),

    # ── Yaw variation ─────────────────────────────────────────────────────────
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.52, 0.0,
                 yaws=[0.0, math.pi/4, -math.pi/4, math.pi/6, -math.pi/6]),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.52, 0.06,
                 yaws=[math.pi/8]*5),
    _mk_scenario([0.35]*5, [-0.16, -0.08, 0.0, 0.08, 0.16], 0.52, -0.06,
                 yaws=[-math.pi/8, math.pi/8, 0.0, math.pi/4, -math.pi/4]),

    # ── Compact cube cluster ──────────────────────────────────────────────────
    _mk_scenario([0.34, 0.35, 0.36, 0.34, 0.36],
                 [-0.06, -0.03, 0.0, 0.03, 0.06], 0.51, 0.0),
    _mk_scenario([0.33, 0.34, 0.35, 0.34, 0.33],
                 [-0.06, -0.03, 0.0, 0.03, 0.06], 0.52, 0.0),

    # ── Asymmetric cube arrangements ──────────────────────────────────────────
    _mk_scenario([0.30, 0.32, 0.34, 0.36, 0.38],
                 [0.0, 0.0, 0.0, 0.0, 0.0], 0.51, 0.0),
    _mk_scenario([0.35, 0.35, 0.35, 0.35, 0.35],
                 [-0.14, 0.14, -0.09, 0.09, 0.0], 0.53, 0.0),
    _mk_scenario([0.32, 0.38, 0.32, 0.38, 0.35],
                 [-0.10, -0.10, 0.10, 0.10, 0.0], 0.51, 0.0),
    _mk_scenario([0.33, 0.33, 0.33, 0.33, 0.33],
                 [-0.14, -0.07, 0.0, 0.07, 0.14], 0.51, -0.04),

    # ── Challenging reach (cubes further forward) ─────────────────────────────
    _mk_scenario([0.38]*5, [-0.14, -0.07, 0.0, 0.07, 0.14], 0.54, 0.0),
    _mk_scenario([0.36]*5, [-0.14, -0.07, 0.0, 0.07, 0.14], 0.53, 0.08),

    # ── Mixed x/y with varied stack ───────────────────────────────────────────
    _mk_scenario([0.31, 0.34, 0.37, 0.34, 0.31],
                 [-0.13, -0.06, 0.0, 0.06, 0.13], 0.50, 0.0),
    _mk_scenario([0.33, 0.36, 0.33, 0.36, 0.34],
                 [-0.10, -0.05, 0.0, 0.05, 0.10], 0.52, 0.05),
    _mk_scenario([0.34, 0.34, 0.34, 0.34, 0.34],
                 [-0.16, -0.08, 0.0, 0.08, 0.16], 0.51, -0.05),

    # ── Scenario 29: maximum x/y separation ──────────────────────────────────
    _mk_scenario([0.30, 0.32, 0.34, 0.36, 0.38],
                 [-0.16, -0.08, 0.0, 0.08, 0.16], 0.54, 0.0),
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
            single_id = env_ids[local_i:local_i + 1]
            cube.write_root_pose_to_sim(pose, env_ids=single_id)
            cube.write_root_velocity_to_sim(velocity[local_i:local_i + 1], env_ids=single_id)
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
