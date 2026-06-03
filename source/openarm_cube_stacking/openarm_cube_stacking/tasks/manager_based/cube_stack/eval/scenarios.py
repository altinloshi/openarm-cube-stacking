from __future__ import annotations

"""Deterministic tournament scenarios for OpenArm cube stacking.

Provides ``NUM_SCENARIOS`` (default 30) fully-deterministic, reproducible
scenarios. Each scenario fixes:

* five cube positions on the tabletop (local frame; ``env_origins`` added at
  reset time),
* an optional per-cube yaw,
* a stack-target XY on the tabletop.

The robot/table placement is shared across all scenarios and comes from
``tabletop.py`` (robot mounted on the table, cubes on the same surface).

Scenarios are generated once with a fixed RNG seed so they are identical across
runs and machines. The eval environment assigns scenario ``env_id % NUM_SCENARIOS``
to each environment.
"""

import math
import random

import torch

from ..tabletop import (
    CUBE_SIZE,
    CUBE_TABLE_Z,
    DEFAULT_STACK_BASE_LOCAL_POS,
    NUM_CUBES,
)

NUM_SCENARIOS: int = 30

# Reachable tabletop region for cube spawns (local frame, robot base at ~x=0.55).
_CUBE_X_RANGE: tuple[float, float] = (0.24, 0.42)
_CUBE_Y_RANGE: tuple[float, float] = (-0.24, 0.24)
# Stack-base region, kept reachable and clear of the spawn row.
_STACK_X_RANGE: tuple[float, float] = (0.40, 0.50)
_STACK_Y_RANGE: tuple[float, float] = (-0.12, 0.12)
# Minimum centre-to-centre separation so cubes / stack base do not overlap.
_MIN_SEPARATION: float = CUBE_SIZE * 1.6
# Optional small cube yaw range (rad).
_YAW_RANGE: tuple[float, float] = (-0.4, 0.4)

# Fixed seed => reproducible scenarios.
_SEED: int = 20260603


def _sample_non_overlapping(rng: random.Random) -> tuple[list[tuple[float, float, float]], tuple[float, float]]:
    """Sample one scenario: 5 non-overlapping cube XY + a separated stack XY."""
    placed: list[tuple[float, float]] = []

    def _far_enough(x: float, y: float) -> bool:
        return all(math.hypot(x - px, y - py) >= _MIN_SEPARATION for px, py in placed)

    # Stack base first so cubes avoid it as well.
    while True:
        sx = rng.uniform(*_STACK_X_RANGE)
        sy = rng.uniform(*_STACK_Y_RANGE)
        break
    placed.append((sx, sy))

    cubes: list[tuple[float, float, float]] = []
    for _ in range(NUM_CUBES):
        for _attempt in range(2000):
            x = rng.uniform(*_CUBE_X_RANGE)
            y = rng.uniform(*_CUBE_Y_RANGE)
            if _far_enough(x, y):
                placed.append((x, y))
                cubes.append((x, y, CUBE_TABLE_Z))
                break
        else:
            # Fallback: accept the last sample even if tight (keeps generation finite).
            cubes.append((x, y, CUBE_TABLE_Z))
            placed.append((x, y))
    return cubes, (sx, sy)


def _generate() -> tuple[list[list[tuple[float, float, float]]], list[list[float]], list[tuple[float, float, float]]]:
    """Generate all scenarios as plain Python lists (deterministic)."""
    rng = random.Random(_SEED)
    all_cube_pos: list[list[tuple[float, float, float]]] = []
    all_cube_yaw: list[list[float]] = []
    all_stack_base: list[tuple[float, float, float]] = []

    for _ in range(NUM_SCENARIOS):
        cubes, (sx, sy) = _sample_non_overlapping(rng)
        yaws = [rng.uniform(*_YAW_RANGE) for _ in range(NUM_CUBES)]
        all_cube_pos.append(cubes)
        all_cube_yaw.append(yaws)
        # Stack base slot-0 centre rests on the tabletop.
        all_stack_base.append((sx, sy, CUBE_TABLE_Z))

    return all_cube_pos, all_cube_yaw, all_stack_base


# Bake scenarios at import time.
_CUBE_POS, _CUBE_YAW, _STACK_BASE = _generate()


def get_scenario_tensors(device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scenario tensors on ``device``.

    Returns:
        cube_pos_local:   ``(NUM_SCENARIOS, NUM_CUBES, 3)`` cube centres (local).
        cube_yaw:         ``(NUM_SCENARIOS, NUM_CUBES)`` cube yaw (rad).
        stack_base_local: ``(NUM_SCENARIOS, 3)`` stack-base centre (local).
    """
    cube_pos = torch.tensor(_CUBE_POS, dtype=torch.float32, device=device)
    cube_yaw = torch.tensor(_CUBE_YAW, dtype=torch.float32, device=device)
    stack_base = torch.tensor(_STACK_BASE, dtype=torch.float32, device=device)
    return cube_pos, cube_yaw, stack_base


def describe() -> dict:
    """Return a JSON-serialisable description of the scenarios (for reports)."""
    return {
        "num_scenarios": NUM_SCENARIOS,
        "num_cubes": NUM_CUBES,
        "cube_size": CUBE_SIZE,
        "default_stack_base_local": list(DEFAULT_STACK_BASE_LOCAL_POS),
        "seed": _SEED,
    }
