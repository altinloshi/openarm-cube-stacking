from __future__ import annotations

import math
from enum import IntEnum

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude, quat_from_euler_xyz

from ..tabletop_scene import CUBE_SIZE, CUBE_TABLE_Z, NUM_CUBES, TABLE_TOP_Z


class Stage(IntEnum):
    PRE_GRASP = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    MOVE_ABOVE_STACK = 4
    LOWER_TO_STACK = 5
    RELEASE = 6
    RETRACT = 7
    NEXT_CUBE = 8
    DONE = 9


STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in Stage)
_STAGE_GRIP = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def _grasp_yaw(cube_quat_w: torch.Tensor) -> torch.Tensor:
    """Snap cube yaw to the nearest face-symmetric 90-degree grasp yaw."""
    _roll, _pitch, yaw = euler_xyz_from_quat(cube_quat_w)
    half_pi = 0.5 * math.pi
    return (yaw + 0.25 * math.pi) % half_pi - 0.25 * math.pi


class ClassicalStackPlanner:
    """Vectorized static-endpoint planner for five-cube tabletop stacking."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        num_cubes: int = NUM_CUBES,
        ee_tcp_offset_z: float = 0.0,
        pre_grasp_height: float = 0.12,
        lift_height: float = 0.16,
        retract_height: float = 0.14,
        grasp_z_offset: float = 0.015,
        release_z_offset: float = 0.035,
        pos_tol: float = 0.025,
        ang_tol: float = 0.25,
        pos_tol_grasp: float = 0.045,
        ang_tol_grasp: float = 0.50,
        min_stage_dur: float = 0.20,
        grasp_hold_s: float = 0.45,
        release_hold_s: float = 0.35,
        next_cube_hold_s: float = 0.15,
        max_retries: int = 3,
        eval_mode: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.num_cubes = num_cubes
        self.device = device
        self.ee_tcp_offset_z = ee_tcp_offset_z
        self.pre_grasp_height = pre_grasp_height
        self.lift_height = lift_height
        self.retract_height = retract_height
        self.grasp_z_offset = grasp_z_offset
        self.release_z_offset = release_z_offset
        self.min_stage_dur = min_stage_dur
        self.grasp_hold_s = grasp_hold_s
        self.release_hold_s = release_hold_s
        self.next_cube_hold_s = next_cube_hold_s
        self.max_retries = max_retries
        self.eval_mode = eval_mode

        self._stage = torch.full((num_envs,), int(Stage.PRE_GRASP), dtype=torch.long, device=device)
        self._current_cube_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._elapsed = torch.zeros(num_envs, device=device)
        self._retry_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._grasp_retries_total = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._cube_failed = torch.zeros(num_envs, num_cubes, dtype=torch.bool, device=device)
        self._episode_failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._yaw = torch.zeros(num_envs, device=device)

        self._target_pos = torch.zeros(num_envs, 3, device=device)
        self._target_quat = torch.zeros(num_envs, 4, device=device)
        self._target_quat[:, 0] = 1.0
        self._pos_err = torch.zeros(num_envs, device=device)
        self._ang_err = torch.zeros(num_envs, device=device)
        self._track_ok = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._stage_changed = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self._grip_table = _STAGE_GRIP.to(device=device, dtype=torch.float32)
        pos_tols = [pos_tol, pos_tol, pos_tol_grasp, pos_tol, pos_tol, pos_tol, pos_tol_grasp, pos_tol, pos_tol, pos_tol]
        ang_tols = [ang_tol, ang_tol, ang_tol_grasp, ang_tol, ang_tol, ang_tol, ang_tol_grasp, ang_tol, ang_tol, ang_tol]
        self._pos_tol_table = torch.tensor(pos_tols, dtype=torch.float32, device=device)
        self._ang_tol_table = torch.tensor(ang_tols, dtype=torch.float32, device=device)

    @property
    def stage(self) -> torch.Tensor:
        return self._stage

    @property
    def current_cube_idx(self) -> torch.Tensor:
        return self._current_cube_idx

    @property
    def retry_count(self) -> torch.Tensor:
        return self._retry_count

    @property
    def total_retries(self) -> torch.Tensor:
        return self._grasp_retries_total

    @property
    def episode_failed(self) -> torch.Tensor:
        return self._episode_failed

    @property
    def cube_failed(self) -> torch.Tensor:
        return self._cube_failed

    def is_done(self) -> torch.Tensor:
        return self._stage == int(Stage.DONE)

    def reset(self, env_ids: torch.Tensor, cube_quat_w: torch.Tensor | None = None) -> None:
        """Reset selected envs to the first cube and PRE_GRASP."""
        if env_ids.numel() == 0:
            return
        self._stage[env_ids] = int(Stage.PRE_GRASP)
        self._current_cube_idx[env_ids] = 0
        self._elapsed[env_ids] = 0.0
        self._retry_count[env_ids] = 0
        self._grasp_retries_total[env_ids] = 0
        self._cube_failed[env_ids] = False
        self._episode_failed[env_ids] = False
        self._target_pos[env_ids] = 0.0
        self._target_quat[env_ids] = 0.0
        self._target_quat[env_ids, 0] = 1.0
        if cube_quat_w is not None:
            self._yaw[env_ids] = _grasp_yaw(cube_quat_w[env_ids])
        else:
            self._yaw[env_ids] = 0.0

    def step(
        self,
        cube_pos_w: torch.Tensor,
        cube_quat_w: torch.Tensor,
        stack_base_pos_w: torch.Tensor,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance planner and return world-frame target pose, grip, cube index, and stage."""
        old_stage = self._stage.clone()
        target_pos, target_quat = self._end_pose(cube_pos_w, cube_quat_w, stack_base_pos_w)
        self._elapsed += dt

        pos_err = torch.norm(ee_pos_w - target_pos, dim=-1)
        ang_err = quat_error_magnitude(ee_quat_w, target_quat)
        track_ok = (pos_err < self._pos_tol_table[self._stage]) & (ang_err < self._ang_tol_table[self._stage])
        self._pos_err.copy_(pos_err)
        self._ang_err.copy_(ang_err)
        self._track_ok.copy_(track_ok)

        dwell = torch.full_like(self._elapsed, self.min_stage_dur)
        dwell = torch.where(self._stage == int(Stage.GRASP), dwell + self.grasp_hold_s, dwell)
        dwell = torch.where(self._stage == int(Stage.RELEASE), dwell + self.release_hold_s, dwell)
        dwell = torch.where(self._stage == int(Stage.NEXT_CUBE), self.next_cube_hold_s * torch.ones_like(dwell), dwell)
        can_advance = (self._elapsed >= dwell) & track_ok & (self._stage != int(Stage.DONE))

        lift_miss = self._handle_lift_miss(cube_pos_w)
        self._advance_stages(can_advance & ~lift_miss)

        self._stage_changed = old_stage != self._stage
        self._target_pos.copy_(target_pos)
        self._target_quat.copy_(target_quat)
        grip = self._grip_table[self._stage].unsqueeze(-1)
        return target_pos, target_quat, grip, self._current_cube_idx.clone(), self._stage.clone()

    def _handle_lift_miss(self, cube_pos_w: torch.Tensor) -> torch.Tensor:
        """Retry a cube if it was not lifted after the LIFT dwell window."""
        in_lift = self._stage == int(Stage.LIFT)
        lift_ready = self._elapsed >= (self.min_stage_dur + 0.15)
        expected_lift_z = TABLE_TOP_Z + CUBE_SIZE + 0.04
        missed = in_lift & lift_ready & (cube_pos_w[:, 2] < expected_lift_z)
        if not missed.any():
            return missed

        can_retry = missed & (self._retry_count < self.max_retries)
        if can_retry.any():
            self._retry_count[can_retry] += 1
            self._grasp_retries_total[can_retry] += 1
            self._stage[can_retry] = int(Stage.PRE_GRASP)
            self._elapsed[can_retry] = 0.0

        exceeded = missed & ~can_retry
        if exceeded.any():
            cube_ids = self._current_cube_idx[exceeded]
            self._cube_failed[exceeded, cube_ids] = True
            self._episode_failed[exceeded] = self.eval_mode
            if self.eval_mode:
                self._stage[exceeded] = int(Stage.DONE)
            else:
                self._stage[exceeded] = int(Stage.NEXT_CUBE)
            self._elapsed[exceeded] = 0.0
        return missed

    def _advance_stages(self, can_advance: torch.Tensor) -> None:
        if not can_advance.any():
            return
        next_stage = torch.clamp(self._stage + 1, max=int(Stage.DONE))
        self._stage = torch.where(can_advance, next_stage, self._stage)
        self._elapsed = torch.where(can_advance, torch.zeros_like(self._elapsed), self._elapsed)

        next_cube = can_advance & (next_stage == int(Stage.NEXT_CUBE))
        if next_cube.any():
            last_cube = self._current_cube_idx >= (self.num_cubes - 1)
            finish = next_cube & last_cube
            continue_envs = next_cube & ~last_cube
            self._stage[finish] = int(Stage.DONE)
            self._current_cube_idx[continue_envs] += 1
            self._retry_count[continue_envs] = 0
            self._stage[continue_envs] = int(Stage.PRE_GRASP)

    def _end_pose(
        self,
        cube_pos_w: torch.Tensor,
        cube_quat_w: torch.Tensor,
        stack_base_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the static endpoint pose for each current stage."""
        current_i = self._current_cube_idx.to(cube_pos_w.dtype)
        stack_center_z = CUBE_TABLE_Z + current_i * CUBE_SIZE
        carry_z = TABLE_TOP_Z + self.lift_height

        z_pre = cube_pos_w[:, 2] + self.pre_grasp_height + self.ee_tcp_offset_z
        z_grasp = cube_pos_w[:, 2] + self.grasp_z_offset + self.ee_tcp_offset_z
        z_lift = torch.full_like(z_pre, carry_z + self.ee_tcp_offset_z)
        z_stack_above = torch.full_like(z_pre, carry_z + self.ee_tcp_offset_z)
        z_lower = stack_center_z + self.grasp_z_offset + self.ee_tcp_offset_z
        z_release = stack_center_z + self.release_z_offset + self.ee_tcp_offset_z
        z_retract = stack_center_z + self.retract_height + self.ee_tcp_offset_z

        z_table = torch.stack(
            [z_pre, z_grasp, z_grasp, z_lift, z_stack_above, z_lower, z_release, z_retract, z_retract, z_retract],
            dim=1,
        )
        target_z = z_table.gather(1, self._stage.unsqueeze(-1)).squeeze(-1)

        use_stack_xy = self._stage >= int(Stage.MOVE_ABOVE_STACK)
        target_x = torch.where(use_stack_xy, stack_base_pos_w[:, 0], cube_pos_w[:, 0])
        target_y = torch.where(use_stack_xy, stack_base_pos_w[:, 1], cube_pos_w[:, 1])
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)

        update_yaw = self._stage == int(Stage.PRE_GRASP)
        if update_yaw.any():
            self._yaw[update_yaw] = _grasp_yaw(cube_quat_w[update_yaw])
        target_quat = quat_from_euler_xyz(
            torch.zeros_like(self._yaw),
            torch.full_like(self._yaw, math.pi),
            self._yaw,
        )
        return target_pos, target_quat
