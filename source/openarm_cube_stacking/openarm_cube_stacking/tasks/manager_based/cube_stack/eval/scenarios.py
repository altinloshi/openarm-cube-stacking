from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..tabletop_scene import (
    CUBE_TABLE_Z,
    DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    DEFAULT_STACK_BASE_LOCAL_POS,
    NUM_CUBES,
    ROBOT_BASE_ON_TABLE_POS,
    TABLE_CENTER,
    TABLE_TOP_Z,
)


@dataclass(frozen=True)
class StackScenario:
    cube_positions: tuple[tuple[float, float, float], ...]
    stack_target_xy: tuple[float, float]
    cube_yaws: tuple[float, ...]
    robot_base_pos: tuple[float, float, float] = ROBOT_BASE_ON_TABLE_POS
    table_center: tuple[float, float, float] = TABLE_CENTER

    @property
    def stack_base_xyz(self) -> tuple[float, float, float]:
        return (self.stack_target_xy[0], self.stack_target_xy[1], CUBE_TABLE_Z)


def _scenario(idx: int) -> StackScenario:
    row = idx // 5
    col = idx % 5
    x_shift = (col - 2) * 0.015
    y_shift = (row - 2.5) * 0.012
    base_positions = []
    for cube_id, pos in enumerate(DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS):
        stagger = (cube_id - 2) * 0.004
        base_positions.append((pos[0] + x_shift, pos[1] + y_shift + stagger, CUBE_TABLE_Z))
    stack_x = DEFAULT_STACK_BASE_LOCAL_POS[0] + ((idx % 3) - 1) * 0.025
    stack_y = DEFAULT_STACK_BASE_LOCAL_POS[1] + (((idx // 3) % 3) - 1) * 0.025
    yaws = tuple(((idx + cube_id) % 4) * (math.pi / 2.0) for cube_id in range(NUM_CUBES))
    return StackScenario(tuple(base_positions), (stack_x, stack_y), yaws)


SCENARIOS: tuple[StackScenario, ...] = tuple(_scenario(i) for i in range(30))


def get_scenario(index: int) -> StackScenario:
    return SCENARIOS[index % len(SCENARIOS)]


def scenario_tensors(env_ids: torch.Tensor, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local cube poses and stack bases for ``scenario = env_id % 30``."""
    cube_pos = torch.zeros(env_ids.numel(), NUM_CUBES, 3, device=device)
    cube_quat = torch.zeros(env_ids.numel(), NUM_CUBES, 4, device=device)
    stack_base = torch.zeros(env_ids.numel(), 3, device=device)
    for out_i, env_id in enumerate(env_ids.tolist()):
        scenario = get_scenario(env_id)
        cube_pos[out_i] = torch.tensor(scenario.cube_positions, dtype=torch.float32, device=device)
        stack_base[out_i] = torch.tensor(scenario.stack_base_xyz, dtype=torch.float32, device=device)
        for cube_id, yaw in enumerate(scenario.cube_yaws):
            half = 0.5 * yaw
            cube_quat[out_i, cube_id] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=device)
    return cube_pos, cube_quat, stack_base
