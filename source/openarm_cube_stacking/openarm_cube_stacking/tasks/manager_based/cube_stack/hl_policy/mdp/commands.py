from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from ...tabletop_scene import (
    CUBE_SIZE,
    CUBE_TABLE_Z,
    DEFAULT_STACK_BASE_LOCAL_POS,
    NUM_CUBES,
    OPENARM_EE_BODY,
    TABLE_TOP_Z,
)
from ..classical_stack_planner import ClassicalStackPlanner, STAGE_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LOG = logging.getLogger(__name__)


def make_stack_marker_cfg(cube_size: float = CUBE_SIZE, thickness: float = 0.003) -> VisualizationMarkersCfg:
    """Flat marker showing the stack footprint on the tabletop."""
    return VisualizationMarkersCfg(
        markers={
            "pad": sim_utils.CuboidCfg(
                size=(cube_size, cube_size, thickness),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.85, 0.30),
                    emissive_color=(0.0, 0.25, 0.08),
                    opacity=0.60,
                ),
            )
        }
    ).replace(prim_path="/Visuals/OpenArm/StackTarget")


_TARGET_FRAME_MARKER_CFG = FRAME_MARKER_CFG.copy()
_TARGET_FRAME_MARKER_CFG.prim_path = "/Visuals/OpenArm/PlannerTargetFrame"
_TARGET_FRAME_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


class HLPoseCommand(CommandTerm):
    """Planner-driven EE pose command with LL-compatible ``ee_pose`` format."""

    cfg: "HLPoseCommandCfg"

    def __init__(self, cfg: "HLPoseCommandCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_name]
        self._body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self._cubes: list[RigidObject] = [env.scene[name] for name in cfg.cube_names]

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self._grip_command = torch.zeros(self.num_envs, 1, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat_w[:, 0] = 1.0
        self.stack_base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_stack_base(torch.arange(self.num_envs, device=self.device))

        self.planner = ClassicalStackPlanner(
            num_envs=self.num_envs,
            device=self.device,
            num_cubes=len(self._cubes),
            pre_grasp_height=cfg.pre_grasp_height,
            lift_height=cfg.lift_height,
            retract_height=cfg.retract_height,
            grasp_z_offset=cfg.grasp_z_offset,
            release_z_offset=cfg.release_z_offset,
            pos_tol=cfg.pos_tol,
            ang_tol=cfg.ang_tol,
            pos_tol_grasp=cfg.pos_tol_grasp,
            ang_tol_grasp=cfg.ang_tol_grasp,
            min_stage_dur=cfg.min_stage_dur,
            grasp_hold_s=cfg.grasp_hold_s,
            release_hold_s=cfg.release_hold_s,
            max_retries=cfg.max_retries,
            eval_mode=cfg.eval_mode,
        )

        self.metrics["stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["current_cube_idx"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["retry_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["total_retries"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["track_ok"] = torch.zeros(self.num_envs, device=self.device)

        self._marker_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._marker_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._marker_quat_w[:, 0] = 1.0
        self._step_count = 0
        if cfg.enable_log and not _LOG.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _init_stack_base(self, env_ids: torch.Tensor) -> None:
        local = torch.tensor(self.cfg.stack_base_default, dtype=torch.float32, device=self.device).repeat(env_ids.numel(), 1)
        self.stack_base_pos_w[env_ids] = self._env.scene.env_origins[env_ids] + local

    def set_stack_base_from_local(self, env_ids: torch.Tensor, stack_base_local_xyz: torch.Tensor) -> None:
        """Set deterministic stack base positions from scenario-local XYZ values."""
        self.stack_base_pos_w[env_ids] = self._env.scene.env_origins[env_ids] + stack_base_local_xyz
        self._env.stack_base_pos_w = getattr(self._env, "stack_base_pos_w", torch.zeros_like(self.stack_base_pos_w))
        self._env.stack_base_pos_w[env_ids] = self.stack_base_pos_w[env_ids]

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        first_cube_quat = self._cubes[0].data.root_quat_w
        self.planner.reset(ids, cube_quat_w=first_cube_quat)
        self._grip_command[ids] = 0.0
        if self.cfg.enable_log:
            for env_id in ids.tolist():
                _LOG.info("[HL] reset env=%d stack_base=%s", env_id, _fmt_xyz(self.stack_base_pos_w[env_id]))

    def _update_command(self) -> None:
        ee_pos_w = self.robot.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._body_idx]
        arange = torch.arange(self.num_envs, device=self.device)
        cube_idx = self.planner.current_cube_idx.clamp(max=len(self._cubes) - 1)
        cube_pos_all = torch.stack([cube.data.root_pos_w for cube in self._cubes], dim=1)
        cube_quat_all = torch.stack([cube.data.root_quat_w for cube in self._cubes], dim=1)
        cube_pos = cube_pos_all[arange, cube_idx]
        cube_quat = cube_quat_all[arange, cube_idx]

        target_pos_w, target_quat_w, grip, _cube_idx, _stage = self.planner.step(
            cube_pos_w=cube_pos,
            cube_quat_w=cube_quat,
            stack_base_pos_w=self.stack_base_pos_w,
            ee_pos_w=ee_pos_w,
            ee_quat_w=ee_quat_w,
            dt=self._env.step_dt,
        )

        target_pos_b, target_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, target_pos_w, target_quat_w
        )
        self.pose_command_b[:, :3] = target_pos_b
        self.pose_command_b[:, 3:] = target_quat_b
        self._grip_command.copy_(grip)
        self._target_pos_w.copy_(target_pos_w)
        self._target_quat_w.copy_(target_quat_w)
        if self.cfg.enable_log and self._step_count % self.cfg.log_interval == 0:
            self._log_status()
        self._step_count += 1

    def _update_metrics(self) -> None:
        planner = self.planner
        self.metrics["stage"] = planner.stage.float()
        self.metrics["current_cube_idx"] = planner.current_cube_idx.float()
        self.metrics["retry_count"] = planner.retry_count.float()
        self.metrics["total_retries"] = planner.total_retries.float()
        self.metrics["position_error"] = planner._pos_err
        self.metrics["orientation_error"] = planner._ang_err
        self.metrics["track_ok"] = planner._track_ok.float()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "target_frame_visualizer"):
                self.target_frame_visualizer = VisualizationMarkers(_TARGET_FRAME_MARKER_CFG)
                self.stack_marker_visualizer = VisualizationMarkers(self.cfg.stack_marker_visualizer_cfg)
            self.target_frame_visualizer.set_visibility(True)
            self.stack_marker_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_frame_visualizer"):
                self.target_frame_visualizer.set_visibility(False)
                self.stack_marker_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not hasattr(self, "target_frame_visualizer"):
            return
        self.target_frame_visualizer.visualize(self._target_pos_w, self._target_quat_w)
        self._marker_pos_w[:, :2] = self.stack_base_pos_w[:, :2]
        self._marker_pos_w[:, 2] = TABLE_TOP_Z + 0.002
        self.stack_marker_visualizer.visualize(self._marker_pos_w, self._marker_quat_w)

    def _log_status(self) -> None:
        ids = range(self.num_envs) if self.cfg.log_env_id < 0 else [min(self.cfg.log_env_id, self.num_envs - 1)]
        for env_id in ids:
            stage = int(self.planner.stage[env_id].item())
            _LOG.info(
                "[HL] env=%d cube=%d stage=%s target=%s grip=%.0f retries=%d",
                env_id,
                int(self.planner.current_cube_idx[env_id].item()),
                STAGE_NAMES[stage],
                _fmt_xyz(self._target_pos_w[env_id]),
                self._grip_command[env_id, 0].item(),
                int(self.planner.total_retries[env_id].item()),
            )


def _fmt_xyz(v: torch.Tensor) -> str:
    return f"({v[0].item():.3f}, {v[1].item():.3f}, {v[2].item():.3f})"


@configclass
class HLPoseCommandCfg(CommandTermCfg):
    """Configuration for the classical stack planner command."""

    class_type: type = HLPoseCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = True
    robot_name: str = "robot"
    body_name: str = OPENARM_EE_BODY
    cube_names: list[str] = [f"cube_{i}" for i in range(NUM_CUBES)]
    stack_base_default: tuple[float, float, float] = DEFAULT_STACK_BASE_LOCAL_POS
    pre_grasp_height: float = 0.12
    lift_height: float = 0.16
    retract_height: float = 0.14
    grasp_z_offset: float = 0.015
    release_z_offset: float = 0.035
    pos_tol: float = 0.025
    ang_tol: float = 0.25
    pos_tol_grasp: float = 0.045
    ang_tol_grasp: float = 0.50
    min_stage_dur: float = 0.20
    grasp_hold_s: float = 0.45
    release_hold_s: float = 0.35
    max_retries: int = 3
    eval_mode: bool = False
    enable_log: bool = False
    log_interval: int = 120
    log_env_id: int = 0
    stack_marker_visualizer_cfg = make_stack_marker_cfg()


class HLGripCommand(CommandTerm):
    """Mirror the planner grip command into LL-compatible ``grip_cmd``."""

    cfg: "HLGripCommandCfg"

    def __init__(self, cfg: "HLGripCommandCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self._grip_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._grip_command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self._grip_command[env_ids] = 0.0

    def _update_command(self) -> None:
        pose_term: HLPoseCommand = self._env.command_manager.get_term(self.cfg.pose_cmd_name)
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
