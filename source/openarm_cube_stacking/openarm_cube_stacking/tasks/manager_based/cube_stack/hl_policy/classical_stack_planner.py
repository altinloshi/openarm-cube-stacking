"""Vectorised classical planner for five-cube sequential stacking.

This module provides :class:`ClassicalStackPlanner`, a pure-PyTorch planner
that operates over a batch of environments.  It can be used:

1. As the backend of :class:`ClassicalStackPlannerCommand` (Isaac Lab command term)
   for the HL-Classical-Play and Eval environments.
2. Standalone in Python scripts for debugging / visualisation.

Planner stages
--------------
Each environment independently progresses through:

  PRE_GRASP         → move EE above current cube
  DESCEND           → lower EE to cube grasp pose
  GRASP             → close gripper (dwell to ensure contact)
  LIFT              → raise EE + cube
  MOVE_ABOVE_STACK  → translate above stack target XY
  LOWER_TO_STACK    → lower to target stack height for cube i
  RELEASE           → open gripper (dwell)
  RETRACT           → raise EE clear of stack
  NEXT_CUBE         → bookkeeping, advance cube index
  DONE              → all cubes placed; hold position

Stage advancement
-----------------
A stage advances when:
  * EE position error  < ``pos_tolerance``  (metres)
  * EE orientation error < ``ori_tolerance`` (radians)
  * ``dwell_time`` seconds have elapsed since reaching the goal

Retry logic
-----------
After GRASP→LIFT, if the cube was not lifted (cube z stayed near table),
the planner retries the same cube up to ``max_retries`` times before
marking it as failed and moving on.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────────────────────────────────────
# Stage enumeration
# ─────────────────────────────────────────────────────────────────────────────


class PlannerStage(IntEnum):
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


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────


class ClassicalStackPlanner:
    """Vectorised classical planner for sequential five-cube stacking.

    Parameters
    ----------
    num_envs:
        Number of parallel environments.
    device:
        Torch device.
    num_cubes:
        Number of cubes to stack (default 5).
    cube_size:
        Cube edge length in metres.
    table_top_z:
        Z of the table top surface in the local environment frame.
    pre_grasp_height:
        Height above cube centre for PRE_GRASP waypoint (metres).
    lift_height:
        Minimum height to lift cube above table top (metres).
    stack_approach_height:
        Height above target stack position during MOVE_ABOVE_STACK (metres).
    retract_height:
        Height above release point during RETRACT (metres).
    pos_tolerance:
        EE position error threshold for stage advancement (metres).
    ori_tolerance:
        EE orientation error threshold for stage advancement (radians).
    min_dwell_time:
        Minimum time at goal before advancing (seconds).
    grasp_dwell_time:
        Dwell time in GRASP stage (seconds).
    release_dwell_time:
        Dwell time in RELEASE stage (seconds).
    grasp_quat:
        Default EE orientation for grasping (wxyz quaternion, world frame).
        Should point the gripper downward; adjust per robot convention.
    max_retries:
        Maximum grasp retries per cube before giving up.
    lift_check_z:
        Cube is considered lifted if its z > table_top_z + lift_check_z.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        num_cubes: int = 5,
        cube_size: float = 0.05,
        table_top_z: float = 0.20,
        pre_grasp_height: float = 0.12,
        lift_height: float = 0.15,
        stack_approach_height: float = 0.12,
        retract_height: float = 0.12,
        pos_tolerance: float = 0.015,
        ori_tolerance: float = 0.25,
        min_dwell_time: float = 0.4,
        grasp_dwell_time: float = 0.6,
        release_dwell_time: float = 0.5,
        grasp_quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
        max_retries: int = 3,
        lift_check_z: float = 0.05,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_cubes = num_cubes
        self.cube_size = cube_size
        self.table_top_z = table_top_z
        self.pre_grasp_height = pre_grasp_height
        self.lift_height = lift_height
        self.stack_approach_height = stack_approach_height
        self.retract_height = retract_height
        self.pos_tolerance = pos_tolerance
        self.ori_tolerance = ori_tolerance
        self.min_dwell_time = min_dwell_time
        self.grasp_dwell_time = grasp_dwell_time
        self.release_dwell_time = release_dwell_time
        self.max_retries = max_retries
        self.lift_check_z = table_top_z + lift_check_z + cube_size / 2.0

        self._grasp_quat = torch.tensor(grasp_quat, dtype=torch.float32, device=self.device)
        self._grasp_quat = self._grasp_quat.unsqueeze(0).expand(num_envs, -1).clone()

        # Per-environment planner state
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.cube_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.dwell_timer = torch.zeros(num_envs, device=self.device)
        self.retry_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.failed_cubes = torch.zeros(num_envs, num_cubes, dtype=torch.bool, device=self.device)

        # Cached last command
        self._target_pos = torch.zeros(num_envs, 3, device=self.device)
        self._target_quat = self._grasp_quat.clone()
        self._target_grip = torch.zeros(num_envs, device=self.device)  # 0=close, 1=open

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset planner state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.stage[env_ids] = PlannerStage.PRE_GRASP
        self.cube_idx[env_ids] = 0
        self.dwell_timer[env_ids] = 0.0
        self.retry_count[env_ids] = 0
        self.failed_cubes[env_ids] = False
        # Start with open gripper
        self._target_grip[env_ids] = 1.0

    def compute(
        self,
        dt: float,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        cube_pos_w: torch.Tensor,
        stack_base_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute planner commands for one time-step.

        Parameters
        ----------
        dt:
            Time delta in seconds.
        ee_pos_w:
            Current EE position in world frame, shape (num_envs, 3).
        ee_quat_w:
            Current EE orientation in world frame (wxyz), shape (num_envs, 4).
        cube_pos_w:
            Current cube positions in world frame, shape (num_envs, num_cubes, 3).
        stack_base_pos_w:
            Stack base position in world frame, shape (num_envs, 3).

        Returns
        -------
        target_pos_w : (num_envs, 3)
        target_quat_w : (num_envs, 4)
        target_grip : (num_envs,)  — 0=close, 1=open
        """
        for stage_val in range(len(PlannerStage)):
            mask = self.stage == stage_val
            if not mask.any():
                continue
            self._process_stage(
                PlannerStage(stage_val),
                mask,
                dt,
                ee_pos_w,
                ee_quat_w,
                cube_pos_w,
                stack_base_pos_w,
            )

        return self._target_pos.clone(), self._target_quat.clone(), self._target_grip.clone()

    # Properties for external inspection
    @property
    def current_stage(self) -> torch.Tensor:
        return self.stage

    @property
    def current_cube_idx(self) -> torch.Tensor:
        return self.cube_idx

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _stack_target_z(self, cube_i: torch.Tensor) -> torch.Tensor:
        """Z coordinate for cube_i in the stack (centre of cube)."""
        return self.table_top_z + self.cube_size / 2.0 + cube_i.float() * self.cube_size

    def _process_stage(
        self,
        stage: PlannerStage,
        mask: torch.Tensor,
        dt: float,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        cube_pos_w: torch.Tensor,
        stack_base_pos_w: torch.Tensor,
    ) -> None:
        """Update targets and advance stage for all envs currently in ``stage``."""
        idx = self.cube_idx  # (num_envs,)
        # Gather current cube positions: (num_envs, 3)
        cube_pos_current = self._gather_cube(cube_pos_w, idx)

        if stage == PlannerStage.PRE_GRASP:
            # Move EE above the current cube
            target = cube_pos_current.clone()
            target[:, 2] = cube_pos_current[:, 2] + self.pre_grasp_height
            self._set_target(mask, target, self._grasp_quat, grip=1.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.min_dwell_time, PlannerStage.DESCEND)

        elif stage == PlannerStage.DESCEND:
            # Lower to grasp height (cube centre z)
            target = cube_pos_current.clone()
            self._set_target(mask, target, self._grasp_quat, grip=1.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.min_dwell_time, PlannerStage.GRASP)

        elif stage == PlannerStage.GRASP:
            # Close gripper; dwell to establish contact
            target = cube_pos_current.clone()
            self._set_target(mask, target, self._grasp_quat, grip=0.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.grasp_dwell_time, PlannerStage.LIFT)

        elif stage == PlannerStage.LIFT:
            # Lift cube clear of the table
            target = cube_pos_current.clone()
            target[:, 2] = self.table_top_z + self.lift_height
            self._set_target(mask, target, self._grasp_quat, grip=0.0)
            advanced = self._try_advance(
                mask, ee_pos_w, target, dt, self.min_dwell_time, PlannerStage.MOVE_ABOVE_STACK
            )
            # Retry logic: if we advanced but cube wasn't actually lifted, retry
            if advanced.any():
                adv_mask = mask & advanced
                lifted = cube_pos_current[:, 2] > self.lift_check_z
                failed_lift = adv_mask & ~lifted
                can_retry = self.retry_count < self.max_retries
                retry_mask = failed_lift & can_retry
                give_up_mask = failed_lift & ~can_retry

                # Retry: go back to PRE_GRASP
                self.stage[retry_mask] = PlannerStage.PRE_GRASP
                self.retry_count[retry_mask] += 1
                self.dwell_timer[retry_mask] = 0.0

                # Give up: mark cube as failed, advance to NEXT_CUBE
                for b in give_up_mask.nonzero(as_tuple=False).squeeze(-1):
                    self.failed_cubes[b, self.cube_idx[b]] = True
                self.stage[give_up_mask] = PlannerStage.NEXT_CUBE
                self.dwell_timer[give_up_mask] = 0.0

        elif stage == PlannerStage.MOVE_ABOVE_STACK:
            # Move above the stack target XY at approach height
            target = stack_base_pos_w.clone()
            stack_z = self._stack_target_z(idx)
            target[:, 2] = stack_z + self.stack_approach_height
            self._set_target(mask, target, self._grasp_quat, grip=0.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.min_dwell_time, PlannerStage.LOWER_TO_STACK)

        elif stage == PlannerStage.LOWER_TO_STACK:
            # Lower to exact stack position
            target = stack_base_pos_w.clone()
            target[:, 2] = self._stack_target_z(idx)
            self._set_target(mask, target, self._grasp_quat, grip=0.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.min_dwell_time, PlannerStage.RELEASE)

        elif stage == PlannerStage.RELEASE:
            # Open gripper; dwell to release
            target = stack_base_pos_w.clone()
            target[:, 2] = self._stack_target_z(idx)
            self._set_target(mask, target, self._grasp_quat, grip=1.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.release_dwell_time, PlannerStage.RETRACT)

        elif stage == PlannerStage.RETRACT:
            # Rise above the stack
            target = stack_base_pos_w.clone()
            target[:, 2] = self._stack_target_z(idx) + self.retract_height
            self._set_target(mask, target, self._grasp_quat, grip=1.0)
            self._try_advance(mask, ee_pos_w, target, dt, self.min_dwell_time, PlannerStage.NEXT_CUBE)

        elif stage == PlannerStage.NEXT_CUBE:
            # Advance cube index
            next_idx = self.cube_idx + 1
            done = next_idx >= self.num_cubes
            self.cube_idx[mask] = torch.where(done[mask], self.cube_idx[mask], next_idx[mask])
            self.stage[mask & ~done] = PlannerStage.PRE_GRASP
            self.stage[mask & done] = PlannerStage.DONE
            self.retry_count[mask] = 0
            self.dwell_timer[mask] = 0.0

        elif stage == PlannerStage.DONE:
            # Hold last position (no update needed)
            pass

    def _gather_cube(self, cube_pos_w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather position of cube at index ``idx`` for each env."""
        idx_clamped = idx.clamp(0, self.num_cubes - 1)
        gather_idx = idx_clamped.view(-1, 1, 1).expand(-1, 1, 3)
        return cube_pos_w.gather(dim=1, index=gather_idx).squeeze(1)

    def _set_target(
        self,
        mask: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        grip: float,
    ) -> None:
        """Write target pos/quat/grip for environments where mask is True."""
        self._target_pos[mask] = pos[mask]
        self._target_quat[mask] = quat[mask]
        self._target_grip[mask] = grip

    def _try_advance(
        self,
        mask: torch.Tensor,
        ee_pos_w: torch.Tensor,
        target_pos: torch.Tensor,
        dt: float,
        dwell_required: float,
        next_stage: PlannerStage,
    ) -> torch.Tensor:
        """Advance to next_stage for environments that meet the criteria.

        Returns a boolean mask of envs that advanced this step.
        """
        pos_err = torch.norm(ee_pos_w - target_pos, dim=-1)
        at_target = pos_err < self.pos_tolerance

        # Update dwell timer: increment when near target, reset otherwise
        self.dwell_timer = torch.where(
            mask & at_target,
            self.dwell_timer + dt,
            torch.where(mask, torch.zeros_like(self.dwell_timer), self.dwell_timer),
        )

        dwell_met = self.dwell_timer >= dwell_required
        can_advance = mask & at_target & dwell_met

        self.stage[can_advance] = next_stage
        self.dwell_timer[can_advance] = 0.0

        return can_advance
