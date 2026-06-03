from __future__ import annotations

"""Vectorised classical stack planner for the OpenArm HL policy.

Mirrors the Franka ``PickPlacePlanner`` but adapted to sequential five-cube
stacking. The planner is a torch-vectorised finite state machine that, for each
environment, picks cubes one at a time and stacks them at a shared stack base,
emitting *static endpoint* end-effector poses (not interpolated trajectories) so
the frozen LL EE-tracking policy sees the same piecewise-constant command
distribution it was trained on.

Stages (plus the absorbing DONE):

    PRE_GRASP -> DESCEND -> GRASP -> LIFT -> MOVE_ABOVE_STACK
        -> LOWER_TO_STACK -> RELEASE -> RETRACT -> NEXT_CUBE -> (PRE_GRASP | DONE)

Gripper per stage (0 = open, 1 = close):

    PRE_GRASP        open
    DESCEND          open
    GRASP            close
    LIFT             close
    MOVE_ABOVE_STACK close
    LOWER_TO_STACK   close
    RELEASE          open
    RETRACT          open
    NEXT_CUBE        open
    DONE             open

Stack-target height for cube ``i`` (cube centre):

    target_z_i = TABLE_TOP_Z + CUBE_SIZE / 2 + i * CUBE_SIZE

All poses are world-frame. Every Z target is expressed for the EE body
(``openarm_hand``); the hand-to-grasp vertical offset ``hand_tcp_offset_z`` lifts
the commanded hand pose so the gripper TCP reaches the cube/stack height.
"""

import math
from enum import IntEnum

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude, quat_from_euler_xyz


class Stage(IntEnum):
    PRE_GRASP = 0         # hover above the current cube, gripper open
    DESCEND = 1           # lower to grasp depth, gripper open
    GRASP = 2             # close gripper and hold
    LIFT = 3              # raise to carry height at the cube XY
    MOVE_ABOVE_STACK = 4  # move to the stack XY at carry height
    LOWER_TO_STACK = 5    # descend to the per-index stack height
    RELEASE = 6           # open gripper at the stack and hold
    RETRACT = 7           # lift clear above the stack, gripper open
    NEXT_CUBE = 8         # transient: advance to the next cube or finish
    DONE = 9              # absorbing: hold at retract height


STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in Stage)

# Gripper command per stage (index = Stage value).
_STAGE_GRIP: list[float] = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def _grasp_yaw(cube_quat: torch.Tensor) -> torch.Tensor:
    """Snap cube yaw to nearest 90-degree face symmetry -> [-pi/4, pi/4]."""
    _, _, yaw = euler_xyz_from_quat(cube_quat)
    half_pi = 0.5 * math.pi
    return (yaw + 0.25 * math.pi) % half_pi - 0.25 * math.pi


class ClassicalStackPlanner:
    """Vectorised sequential cube-stacking planner (static-endpoint commands).

    Each ``step()`` returns ``(end_pos_w, end_quat_w, grip)`` — the endpoint of
    the current stage — so the LL policy tracks a fixed target until the planner
    advances. Stage advancement is gated on LL EE arrival within
    (``pos_tol``, ``ang_tol``) of the endpoint plus a minimum dwell time.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        num_cubes: int = 5,
        cube_size: float = 0.05,
        table_top_z: float = 0.20,
        hand_tcp_offset_z: float = 0.12,  # openarm_hand above grasp point (m); tune to USD TCP
        pre_approach_z: float = 0.12,     # standoff above cube centre (m)
        carry_z: float = 0.18,            # absolute EE carry height above table top (m)
        grasp_z_offset: float = -0.005,   # TCP below cube/stack centre at grasp/lower (m)
        release_z_offset: float = 0.005,  # TCP slightly above stack centre at release (m)
        retract_approach_z: float = 0.14, # standoff above the stack at retract (m)
        stack_yaw: float = 0.0,           # placement yaw (axis-aligned stack)
        pos_tol: float = 0.015,           # PRE_GRASP / RETRACT / DONE (m)
        ang_tol: float = 0.10,            # PRE_GRASP / RETRACT / DONE (rad)
        pos_tol_approach: float = 0.025,  # DESCEND
        ang_tol_approach: float = 0.20,   # DESCEND
        pos_tol_grasp: float = 0.045,     # GRASP / RELEASE (gripper actuation)
        ang_tol_grasp: float = 0.45,
        pos_tol_transport: float = 0.025, # LIFT / MOVE_ABOVE_STACK / LOWER_TO_STACK
        ang_tol_transport: float = 0.25,
        min_stage_dur: float = 0.25,      # minimum dwell per stage (s)
        grasp_hold_s: float = 0.45,       # extra dwell at GRASP for the grip to close
        release_hold_s: float = 0.35,     # extra dwell at RELEASE for the grip to open
        max_retries: int = 3,             # retries on a missed grasp
        abort_on_failed_grasp: bool = False,  # eval-strict: fail episode if retries exhausted
    ) -> None:
        self.num_envs = num_envs
        self.device = device

        self.num_cubes = num_cubes
        self.cube_size = cube_size
        self.table_top_z = table_top_z
        self.H = hand_tcp_offset_z
        self.pre_approach_z = pre_approach_z
        self.carry_z = carry_z
        self.grasp_z_offset = grasp_z_offset
        self.release_z_offset = release_z_offset
        self.retract_approach_z = retract_approach_z
        self.stack_yaw = stack_yaw
        self.min_stage_dur = min_stage_dur
        self.grasp_hold_s = grasp_hold_s
        self.release_hold_s = release_hold_s
        self.max_retries = max_retries
        self.abort_on_failed_grasp = abort_on_failed_grasp

        # Cube counts as "lifted" when its centre rises this far above the table.
        self.min_lift_height = table_top_z + cube_size / 2.0 + 0.04

        N, dev = num_envs, device
        self._stage = torch.full((N,), int(Stage.PRE_GRASP), dtype=torch.long, device=dev)
        self._elapsed = torch.zeros(N, device=dev)
        self._yaw = torch.zeros(N, device=dev)
        self._retry_count = torch.zeros(N, dtype=torch.long, device=dev)
        self._current_cube_idx = torch.zeros(N, dtype=torch.long, device=dev)
        self._target_pos = torch.zeros(N, 3, device=dev)
        self._target_quat = torch.zeros(N, 4, device=dev)
        self._target_quat[:, 0] = 1.0

        # Per-env, per-cube failure flags and per-env episode-failed flag.
        self._cube_failed = torch.zeros(N, num_cubes, dtype=torch.bool, device=dev)
        self._episode_failed = torch.zeros(N, dtype=torch.bool, device=dev)

        self._grip_table = torch.tensor(_STAGE_GRIP, device=dev)

        # Per-stage arrival tolerances (index = Stage value).
        pos_tol_stages = [
            pos_tol,            # PRE_GRASP
            pos_tol_approach,   # DESCEND
            pos_tol_grasp,      # GRASP
            pos_tol_transport,  # LIFT
            pos_tol_transport,  # MOVE_ABOVE_STACK
            pos_tol_transport,  # LOWER_TO_STACK
            pos_tol_grasp,      # RELEASE
            pos_tol,            # RETRACT
            pos_tol,            # NEXT_CUBE
            pos_tol,            # DONE
        ]
        ang_tol_stages = [
            ang_tol,
            ang_tol_approach,
            ang_tol_grasp,
            ang_tol_transport,
            ang_tol_transport,
            ang_tol_transport,
            ang_tol_grasp,
            ang_tol,
            ang_tol,
            ang_tol,
        ]
        self._pos_tol_table = torch.tensor(pos_tol_stages, dtype=torch.float32, device=dev)
        self._ang_tol_table = torch.tensor(ang_tol_stages, dtype=torch.float32, device=dev)

        # Diagnostics (updated every step).
        self._pos_err = torch.zeros(N, device=dev)
        self._ang_err = torch.zeros(N, device=dev)
        self._track_ok = torch.zeros(N, dtype=torch.bool, device=dev)
        self._grasp_miss = torch.zeros(N, dtype=torch.bool, device=dev)
        self._stage_changed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._pos_tol_eff = torch.full((N,), pos_tol, device=dev)
        self._ang_tol_eff = torch.full((N,), ang_tol, device=dev)

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def stage(self) -> torch.Tensor:
        return self._stage

    @property
    def current_cube_idx(self) -> torch.Tensor:
        return self._current_cube_idx

    def is_fully_done(self) -> torch.Tensor:
        """``True`` for envs that have finished (placed all cubes or failed out)."""
        return self._stage == int(Stage.DONE)

    def reset(
        self,
        env_ids: torch.Tensor,
        cube_quat_w: torch.Tensor | None = None,
    ) -> None:
        """Reset selected envs to PRE_GRASP on cube 0."""
        if env_ids.numel() == 0:
            return
        ids = env_ids
        self._stage[ids] = int(Stage.PRE_GRASP)
        self._elapsed[ids] = 0.0
        self._retry_count[ids] = 0
        self._current_cube_idx[ids] = 0
        self._cube_failed[ids] = False
        self._episode_failed[ids] = False
        self._target_pos[ids] = 0.0
        self._target_quat[ids] = 0.0
        self._target_quat[ids, 0] = 1.0
        self._yaw[ids] = _grasp_yaw(cube_quat_w[ids]) if cube_quat_w is not None else 0.0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        cube_pos_w: torch.Tensor,   # (N, 3) current cube centre
        cube_quat_w: torch.Tensor,  # (N, 4) current cube orientation (wxyz)
        stack_target_pos_w: torch.Tensor,  # (N, 3) stack slot centre for current cube
        ee_pos_w: torch.Tensor,     # (N, 3) openarm_hand world position
        ee_quat_w: torch.Tensor,    # (N, 4) openarm_hand world orientation
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the state machine one control step.

        Returns the current stage endpoint ``(end_pos_w, end_quat_w, grip)``.
        """
        end_pos, end_quat = self._end_pose(cube_pos_w, stack_target_pos_w)
        old_stage = self._stage.clone()
        old_cube_idx = self._current_cube_idx.clone()

        self._elapsed += dt

        # Arrival check against the per-stage tolerances.
        pos_err = torch.norm(ee_pos_w - end_pos, dim=-1)
        ang_err = quat_error_magnitude(ee_quat_w, end_quat)
        pos_tol_eff = self._pos_tol_table[self._stage]
        ang_tol_eff = self._ang_tol_table[self._stage]
        track_ok = (pos_err < pos_tol_eff) & (ang_err < ang_tol_eff)

        self._pos_tol_eff.copy_(pos_tol_eff)
        self._ang_tol_eff.copy_(ang_tol_eff)
        self._pos_err.copy_(pos_err)
        self._ang_err.copy_(ang_err)
        self._track_ok.copy_(track_ok)
        self._grasp_miss.fill_(False)

        in_grasp = self._stage == int(Stage.GRASP)
        in_release = self._stage == int(Stage.RELEASE)
        in_next = self._stage == int(Stage.NEXT_CUBE)
        in_done = self._stage == int(Stage.DONE)

        # Standard stages advance on minimum dwell + arrival.
        at_end = (self._elapsed >= self.min_stage_dur) & track_ok
        # GRASP / RELEASE need extra dwell for the gripper to settle.
        grasp_ok = in_grasp & (self._elapsed >= self.min_stage_dur + self.grasp_hold_s) & track_ok
        release_ok = in_release & (self._elapsed >= self.min_stage_dur + self.release_hold_s) & track_ok

        can_advance = at_end.clone()
        can_advance = torch.where(in_grasp, grasp_ok, can_advance)
        can_advance = torch.where(in_release, release_ok, can_advance)
        # NEXT_CUBE is a transient bookkeeping stage; it always proceeds.
        can_advance = torch.where(in_next, torch.ones_like(can_advance), can_advance)
        can_advance &= ~in_done

        # ------------------------------------------------------------------
        # Grasp-miss recovery: if the cube was not lifted while transporting,
        # retry the same cube from PRE_GRASP (up to ``max_retries``).
        # ------------------------------------------------------------------
        transporting = (self._stage == int(Stage.LIFT)) | (self._stage == int(Stage.MOVE_ABOVE_STACK))
        not_lifted = cube_pos_w[:, 2] < self.min_lift_height
        can_retry = self._retry_count < self.max_retries
        grasp_miss = transporting & not_lifted & can_retry
        # Retries exhausted while transporting -> the cube failed.
        grasp_exhausted = transporting & not_lifted & ~can_retry

        if grasp_miss.any():
            self._grasp_miss.copy_(grasp_miss)
            self._retry_count[grasp_miss] += 1
            self._stage[grasp_miss] = int(Stage.PRE_GRASP)
            self._elapsed[grasp_miss] = 0.0
            self._yaw[grasp_miss] = _grasp_yaw(cube_quat_w[grasp_miss])

        if grasp_exhausted.any():
            idx = self._current_cube_idx[grasp_exhausted]
            self._cube_failed[grasp_exhausted, idx] = True
            if self.abort_on_failed_grasp:
                # Eval-strict: end the episode immediately.
                self._episode_failed[grasp_exhausted] = True
                self._stage[grasp_exhausted] = int(Stage.DONE)
                self._elapsed[grasp_exhausted] = 0.0
            else:
                # Lenient: skip the cube and move on.
                self._stage[grasp_exhausted] = int(Stage.NEXT_CUBE)
                self._elapsed[grasp_exhausted] = 0.0

        # Advance non-retrying, non-exhausted envs.
        if can_advance.any():
            adv = can_advance & ~grasp_miss & ~grasp_exhausted
            self._elapsed = torch.where(adv, torch.zeros_like(self._elapsed), self._elapsed)
            self._stage = torch.where(adv, (self._stage + 1).clamp(max=int(Stage.DONE)), self._stage)

        # ------------------------------------------------------------------
        # NEXT_CUBE handling: advance the cube index or finish.
        # ------------------------------------------------------------------
        entered_next = (self._stage == int(Stage.NEXT_CUBE)) & (old_stage != int(Stage.NEXT_CUBE))
        if entered_next.any():
            has_more = self._current_cube_idx < (self.num_cubes - 1)
            advance_cube = entered_next & has_more
            finish = entered_next & ~has_more

            if advance_cube.any():
                self._current_cube_idx[advance_cube] += 1
                self._stage[advance_cube] = int(Stage.PRE_GRASP)
                self._elapsed[advance_cube] = 0.0
                self._retry_count[advance_cube] = 0
                self._yaw[advance_cube] = _grasp_yaw(cube_quat_w[advance_cube])
            if finish.any():
                self._stage[finish] = int(Stage.DONE)
                self._elapsed[finish] = 0.0

        self._stage_changed = (old_stage != self._stage) | (old_cube_idx != self._current_cube_idx)

        self._target_pos.copy_(end_pos)
        self._target_quat.copy_(end_quat)
        return end_pos, end_quat, self._grip_table[self._stage]

    # ------------------------------------------------------------------
    # Endpoint geometry
    # ------------------------------------------------------------------

    def _end_pose(
        self,
        cube_pos_w: torch.Tensor,
        stack_target_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Target (position, quaternion) in world frame for the current stage."""
        H = self.H
        cx, cy, cz = cube_pos_w[:, 0], cube_pos_w[:, 1], cube_pos_w[:, 2]
        sx, sy, sz = stack_target_pos_w[:, 0], stack_target_pos_w[:, 1], stack_target_pos_w[:, 2]

        # Stages from MOVE_ABOVE_STACK onward use the stack XY and placement yaw.
        use_stack = self._stage >= int(Stage.MOVE_ABOVE_STACK)
        yaw = torch.where(use_stack, torch.full_like(self._yaw, self.stack_yaw), self._yaw)

        # EE-body Z target per stage.
        z_pre = cz + self.pre_approach_z + H            # hover above cube
        z_grasp = cz + self.grasp_z_offset + H          # at grasp depth
        z_carry = torch.full_like(cz, self.table_top_z + self.carry_z + H)
        z_lower = sz + self.grasp_z_offset + H          # at stack place depth
        z_release = sz + self.release_z_offset + H      # just above stack centre
        z_retract = sz + self.retract_approach_z + H    # clear above stack

        # Index order matches the Stage enum (0..9).
        z_table = torch.stack(
            [
                z_pre,       # PRE_GRASP
                z_grasp,     # DESCEND
                z_grasp,     # GRASP
                z_carry,     # LIFT
                z_carry,     # MOVE_ABOVE_STACK
                z_lower,     # LOWER_TO_STACK
                z_release,   # RELEASE
                z_retract,   # RETRACT
                z_retract,   # NEXT_CUBE
                z_retract,   # DONE
            ],
            dim=1,
        )
        ez = z_table.gather(1, self._stage.unsqueeze(-1)).squeeze(-1)

        ex = torch.where(use_stack, sx, cx)
        ey = torch.where(use_stack, sy, cy)

        end_pos = torch.stack([ex, ey, ez], dim=-1)
        end_quat = quat_from_euler_xyz(
            torch.zeros_like(yaw),
            torch.full_like(yaw, math.pi),
            yaw,
        )
        return end_pos, end_quat
