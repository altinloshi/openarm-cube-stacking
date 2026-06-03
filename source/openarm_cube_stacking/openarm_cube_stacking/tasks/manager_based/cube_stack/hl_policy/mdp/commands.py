from __future__ import annotations

"""HL cube-stacking command terms.

``HLStackPoseCommand`` drives the ``ee_pose`` command slot: each step it queries
the :class:`ClassicalStackPlanner` for the *endpoint* of the current stage and
writes that static pose (robot-base frame) so the frozen LL policy tracks the
same piecewise-constant command distribution it was trained on. It also stores
the per-stage gripper command, which ``HLGripCommand`` mirrors into the
``grip_cmd`` slot.

The planner picks cubes sequentially (``current_cube_idx``) and stacks them on a
per-environment ``stack_base_xy``. The stack-slot height for cube ``i`` is
``TABLE_TOP_Z + CUBE_SIZE / 2 + i * CUBE_SIZE``.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from ...tabletop import (
    CUBE_NAMES,
    CUBE_SIZE,
    DEFAULT_STACK_BASE_LOCAL_POS,
    TABLE_TOP_Z,
)
from ..classical_stack_planner import STAGE_NAMES, ClassicalStackPlanner

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LOG = logging.getLogger(__name__)


def make_target_marker_cfg() -> VisualizationMarkersCfg:
    """Frame marker visualising the planner's current EE target pose."""
    cfg = FRAME_MARKER_CFG.copy()
    cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
    cfg.prim_path = "/Visuals/HL/planner_target"
    return cfg


HL_TARGET_MARKER_CFG = make_target_marker_cfg()


class HLStackPoseCommand(CommandTerm):
    """EE pose command driven by the :class:`ClassicalStackPlanner`.

    Outputs ``(N, 7)`` in robot-base frame ``[pos | quat(wxyz)]`` so the frozen
    LL policy runs unchanged.
    """

    cfg: HLStackPoseCommandCfg

    def __init__(self, cfg: HLStackPoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.robot_name]
        self._body_idx: int = self.robot.find_bodies(cfg.body_name)[0][0]

        self._cubes: list[RigidObject] = [env.scene[name] for name in cfg.cube_names]
        self._num_cubes = len(self._cubes)

        # Pose command in robot-base frame (identity quaternion to start).
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

        # Per-env stack base in world frame (set at reset; default until then).
        self.stack_base_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_stack_base(torch.arange(self.num_envs, device=self.device))

        self._grip_command = torch.zeros(self.num_envs, 1, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat_w[:, 0] = 1.0

        self.planner = ClassicalStackPlanner(
            num_envs=self.num_envs,
            device=self.device,
            num_cubes=self._num_cubes,
            cube_size=cfg.cube_size,
            table_top_z=cfg.table_top_z,
            hand_tcp_offset_z=cfg.hand_tcp_offset_z,
            pre_approach_z=cfg.pre_approach_z,
            carry_z=cfg.carry_z,
            grasp_z_offset=cfg.grasp_z_offset,
            release_z_offset=cfg.release_z_offset,
            retract_approach_z=cfg.retract_approach_z,
            stack_yaw=cfg.stack_yaw,
            pos_tol=cfg.pos_tol,
            ang_tol=cfg.ang_tol,
            pos_tol_approach=cfg.pos_tol_approach,
            ang_tol_approach=cfg.ang_tol_approach,
            pos_tol_grasp=cfg.pos_tol_grasp,
            ang_tol_grasp=cfg.ang_tol_grasp,
            pos_tol_transport=cfg.pos_tol_transport,
            ang_tol_transport=cfg.ang_tol_transport,
            min_stage_dur=cfg.min_stage_dur,
            grasp_hold_s=cfg.grasp_hold_s,
            release_hold_s=cfg.release_hold_s,
            max_retries=cfg.max_retries,
            abort_on_failed_grasp=cfg.abort_on_failed_grasp,
        )

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["current_cube_idx"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["retry_count"] = torch.zeros(self.num_envs, device=self.device)

        self._step_count = 0
        if cfg.enable_log and not _LOG.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ------------------------------------------------------------------
    # Stack-base management
    # ------------------------------------------------------------------

    def _init_stack_base(self, env_ids: torch.Tensor) -> None:
        base = torch.tensor(self.cfg.stack_base_local, dtype=torch.float32, device=self.device)
        self.stack_base_w[env_ids] = self._env.scene.env_origins[env_ids] + base

    def set_stack_base(self, env_ids: torch.Tensor, stack_base_local: torch.Tensor) -> None:
        """Set per-env stack base from a local-frame ``(N, 3)`` tensor."""
        origins = self._env.scene.env_origins[env_ids]
        self.stack_base_w[env_ids] = origins + stack_base_local

    def _current_cube_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the current cube's world pose, indexed by the planner."""
        idx = self.planner.current_cube_idx
        arange = torch.arange(self.num_envs, device=self.device)
        cube_pos_all = torch.stack([c.data.root_pos_w for c in self._cubes], dim=1)  # (N, M, 3)
        cube_quat_all = torch.stack([c.data.root_quat_w for c in self._cubes], dim=1)  # (N, M, 4)
        return cube_pos_all[arange, idx], cube_quat_all[arange, idx]

    def _stack_target_pos_w(self) -> torch.Tensor:
        """Stack-slot centre (world) for the current cube index per env."""
        idx = self.planner.current_cube_idx.to(torch.float32)
        target = self.stack_base_w.clone()
        # Base slot 0 sits at TABLE_TOP_Z + CUBE_SIZE/2 (already in stack_base z);
        # each subsequent cube is one cube edge higher.
        target[:, 2] = target[:, 2] + idx * self.cfg.cube_size
        return target

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            "HLStackPoseCommand (ClassicalStackPlanner)\n"
            f"\tCubes        : {self.cfg.cube_names}\n"
            f"\tStack base   : {self.cfg.stack_base_local}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        cube0_quat = self._cubes[0].data.root_quat_w
        self.planner.reset(ids, cube_quat_w=cube0_quat)
        self._grip_command[ids] = 0.0

    def _update_command(self) -> None:
        ee_pos_w = self.robot.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._body_idx]

        cube_pos, cube_quat = self._current_cube_pose()
        stack_target = self._stack_target_pos_w()

        end_pos_w, end_quat_w, grip = self.planner.step(
            cube_pos_w=cube_pos,
            cube_quat_w=cube_quat,
            stack_target_pos_w=stack_target,
            ee_pos_w=ee_pos_w,
            ee_quat_w=ee_quat_w,
            dt=self._env.step_dt,
        )

        end_pos_b, end_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, end_pos_w, end_quat_w
        )
        self.pose_command_b[:, :3] = end_pos_b
        self.pose_command_b[:, 3:] = end_quat_b
        self._grip_command[:, 0] = grip
        self._target_pos_w.copy_(end_pos_w)
        self._target_quat_w.copy_(end_quat_w)

        if self.cfg.enable_log:
            self._log_events(ee_pos_w)
        self._step_count += 1

    def _update_metrics(self) -> None:
        ee_pos_w = self.robot.data.body_pos_w[:, self._body_idx]
        p = self.planner
        self.metrics["position_error"] = torch.norm(ee_pos_w - self._target_pos_w, dim=-1)
        self.metrics["orientation_error"] = p._ang_err
        self.metrics["stage"] = p.stage.float()
        self.metrics["current_cube_idx"] = p.current_cube_idx.float()
        self.metrics["retry_count"] = p._retry_count.float()

    def _log_events(self, ee_pos_w: torch.Tensor) -> None:
        p = self.planner
        for i in torch.where(p._stage_changed)[0].tolist():
            stage = int(p.stage[i].item())
            cube_idx = int(p.current_cube_idx[i].item())
            _LOG.info(
                "[HL] env %d cube %d/%d  stage -> %d %s  grip=%.0f",
                i, cube_idx, self._num_cubes - 1, stage, STAGE_NAMES[stage],
                self._grip_command[i, 0].item(),
            )

    # ------------------------------------------------------------------
    # Debug visualisation: planner target frame
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not hasattr(self, "target_visualizer"):
            return
        self.target_visualizer.visualize(self._target_pos_w, self._target_quat_w)


@configclass
class HLStackPoseCommandCfg(CommandTermCfg):
    """Configuration for :class:`HLStackPoseCommand`."""

    class_type: type = HLStackPoseCommand

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = True

    robot_name: str = "robot"
    body_name: str = "openarm_hand"

    # Cube scene-entity names in pick order.
    cube_names: list[str] = list(CUBE_NAMES)

    # Default stack base (local frame); overwritten per-env by the reset event.
    stack_base_local: tuple[float, float, float] = DEFAULT_STACK_BASE_LOCAL_POS

    # Geometry (kept in sync with tabletop.py).
    cube_size: float = CUBE_SIZE
    table_top_z: float = TABLE_TOP_Z

    target_visualizer_cfg: VisualizationMarkersCfg = HL_TARGET_MARKER_CFG

    # ClassicalStackPlanner parameters.
    hand_tcp_offset_z: float = 0.12
    pre_approach_z: float = 0.12
    carry_z: float = 0.18
    grasp_z_offset: float = -0.005
    release_z_offset: float = 0.005
    retract_approach_z: float = 0.14
    stack_yaw: float = 0.0
    pos_tol: float = 0.015
    ang_tol: float = 0.10
    pos_tol_approach: float = 0.025
    ang_tol_approach: float = 0.20
    pos_tol_grasp: float = 0.045
    ang_tol_grasp: float = 0.45
    pos_tol_transport: float = 0.025
    ang_tol_transport: float = 0.25
    min_stage_dur: float = 0.25
    grasp_hold_s: float = 0.45
    release_hold_s: float = 0.35
    max_retries: int = 3
    abort_on_failed_grasp: bool = False

    enable_log: bool = False


class HLGripCommand(CommandTerm):
    """Mirrors the grip value from :class:`HLStackPoseCommand` into ``grip_cmd``."""

    cfg: HLGripCommandCfg

    def __init__(self, cfg: HLGripCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._grip_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._grip_command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self._grip_command[env_ids] = 0.0

    def _update_command(self) -> None:
        pose_term: HLStackPoseCommand = self._env.command_manager.get_term(self.cfg.pose_cmd_name)
        self._grip_command.copy_(pose_term._grip_command)

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class HLGripCommandCfg(CommandTermCfg):
    """Configuration for :class:`HLGripCommand`."""

    class_type: type = HLGripCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    pose_cmd_name: str = "ee_pose"
