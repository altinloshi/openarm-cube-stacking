"""Isaac Lab command term that wraps ClassicalStackPlanner.

ClassicalStackPlannerCommand is a CommandTerm whose output is the planned
target EE pose (pos + quat = 7 dims), matching the output shape of
UniformPoseCommandCfg so the same LL observation vector can be used for
both LL training and HL inference.

The planner also writes ``env.gripper_cmd`` (num_envs, 1) each step so
that the gripper-tracking observation terms work identically to the LL env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from ..classical_stack_planner import ClassicalStackPlanner
from ...openarm_lift_style_scene_cfg import (
    CUBE_NAMES,
    CUBE_SIZE,
    NUM_CUBES,
    TABLE_TOP_Z,
    TABLETOP_STACK_BASE_LOCAL_POS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ClassicalStackPlannerCommand(CommandTerm):
    """Isaac Lab command term backed by the classical stack planner.

    Output shape: (num_envs, 7) = [target_pos(3), target_quat(4)]
    expressed in the robot base frame (same convention as UniformPoseCommandCfg).

    Additionally writes ``env.gripper_cmd`` (num_envs, 1) so that the
    gripper-tracking observation terms work transparently.
    """

    cfg: "ClassicalStackPlannerCommandCfg"

    def __init__(
        self,
        cfg: "ClassicalStackPlannerCommandCfg",
        env: "ManagerBasedRLEnv",
    ) -> None:
        super().__init__(cfg, env)

        self._planner = ClassicalStackPlanner(
            num_envs=env.num_envs,
            device=env.device,
            num_cubes=cfg.num_cubes,
            cube_size=cfg.cube_size,
            table_top_z=cfg.table_top_z,
            pre_grasp_height=cfg.pre_grasp_height,
            lift_height=cfg.lift_height,
            stack_approach_height=cfg.stack_approach_height,
            retract_height=cfg.retract_height,
            pos_tolerance=cfg.pos_tolerance,
            ori_tolerance=cfg.ori_tolerance,
            min_dwell_time=cfg.min_dwell_time,
            grasp_dwell_time=cfg.grasp_dwell_time,
            release_dwell_time=cfg.release_dwell_time,
            grasp_quat=cfg.grasp_quat,
            max_retries=cfg.max_retries,
        )

        # Command buffer: pos(3) + quat(4) = 7, expressed in robot base frame
        self._command = torch.zeros(env.num_envs, 7, device=env.device)
        # Initialise gripper command on env (start open = 1.0)
        env.gripper_cmd = torch.ones((env.num_envs, 1), device=env.device)

        # Stack base local position (env origins added at runtime each step)
        _local = torch.tensor(
            list(cfg.stack_base_local[:3]), dtype=torch.float32, device=env.device
        )
        self._stack_base_local = _local.unsqueeze(0).expand(env.num_envs, -1).clone()

    # ─────────────────────────────────────────────────────────────────────────
    # CommandTerm interface
    # ─────────────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return "ClassicalStackPlannerCommand"

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._planner.reset(env_ids)
        return {}

    def compute(self, dt: float) -> None:
        """Called every env step; runs planner and writes command."""
        env = self._env

        # ── current EE pose in world frame ─────────────────────────────────
        ee_frame: FrameTransformer = env.scene["ee_frame"]
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]    # (N, 3)
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (N, 4)

        # ── cube positions in world frame ───────────────────────────────────
        cube_pos_w = torch.stack(
            [env.scene[name].data.root_pos_w for name in self.cfg.cube_names],
            dim=1,
        )  # (N, num_cubes, 3)

        # ── stack base in world frame ────────────────────────────────────────
        stack_base_w = env.scene.env_origins + self._stack_base_local

        # ── run planner ──────────────────────────────────────────────────────
        target_pos_w, target_quat_w, target_grip = self._planner.compute(
            dt=dt,
            ee_pos_w=ee_pos_w,
            ee_quat_w=ee_quat_w,
            cube_pos_w=cube_pos_w,
            stack_base_pos_w=stack_base_w,
        )

        # ── convert world → robot base frame ─────────────────────────────────
        robot = env.scene["robot"]
        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w

        target_pos_b, target_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w, target_pos_w, target_quat_w
        )

        self._command[:, :3] = target_pos_b
        self._command[:, 3:7] = target_quat_b

        # ── propagate gripper command to env ──────────────────────────────────
        if not hasattr(env, "gripper_cmd") or env.gripper_cmd.shape[0] != env.num_envs:
            env.gripper_cmd = torch.zeros((env.num_envs, 1), device=env.device)
        env.gripper_cmd[:, 0] = target_grip

    # ─────────────────────────────────────────────────────────────────────────
    # Debug visualisation: stack-target tower + current target level
    # ─────────────────────────────────────────────────────────────────────────

    def _stack_target_translations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return marker translations and per-marker prototype indices.

        Marker 0 (small frame) is drawn at every stack level for every env so
        the full target tower is visible.  Marker 1 (larger frame) highlights
        the level of the cube the planner is currently placing.
        """
        env = self._env
        base = env.scene.env_origins + self._stack_base_local  # (N, 3)
        levels = torch.arange(self.cfg.num_cubes, device=env.device, dtype=torch.float32)
        # All tower levels: (N, num_cubes, 3)
        tower = base[:, None, :].expand(-1, self.cfg.num_cubes, -1).clone()
        tower[:, :, 2] = base[:, None, 2] + levels[None, :] * self.cfg.cube_size
        tower = tower.reshape(-1, 3)
        tower_idx = torch.zeros(tower.shape[0], dtype=torch.long, device=env.device)

        # Current target level per env (clamped to the tower height).
        cur = self._planner.cube_idx.clamp(0, self.cfg.num_cubes - 1).float()
        current = base.clone()
        current[:, 2] = base[:, 2] + cur * self.cfg.cube_size
        current_idx = torch.ones(current.shape[0], dtype=torch.long, device=env.device)

        translations = torch.cat([tower, current], dim=0)
        marker_indices = torch.cat([tower_idx, current_idx], dim=0)
        return translations, marker_indices

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_stack_marker"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/StackTarget"
                marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                # Second prototype: a larger frame for the current target level.
                marker_cfg.markers["current"] = marker_cfg.markers["frame"].copy()
                marker_cfg.markers["current"].scale = (0.12, 0.12, 0.12)
                self._stack_marker = VisualizationMarkers(marker_cfg)
            self._stack_marker.set_visibility(True)
        elif hasattr(self, "_stack_marker"):
            self._stack_marker.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not hasattr(self, "_stack_marker"):
            return
        translations, marker_indices = self._stack_target_translations()
        self._stack_marker.visualize(translations=translations, marker_indices=marker_indices)


@configclass
class ClassicalStackPlannerCommandCfg(CommandTermCfg):
    """Configuration for the classical stack planner command term."""

    class_type: type = ClassicalStackPlannerCommand

    # Planner geometry (must match the scene constants)
    num_cubes: int = NUM_CUBES
    cube_names: list = list(CUBE_NAMES)
    cube_size: float = CUBE_SIZE
    table_top_z: float = TABLE_TOP_Z

    # Stack base position in LOCAL env frame (env origin added at runtime)
    stack_base_local: tuple = TABLETOP_STACK_BASE_LOCAL_POS

    # Planner stage tuning
    pre_grasp_height: float = 0.12
    lift_height: float = 0.15
    stack_approach_height: float = 0.12
    retract_height: float = 0.12
    pos_tolerance: float = 0.015
    ori_tolerance: float = 0.25
    min_dwell_time: float = 0.4
    grasp_dwell_time: float = 0.6
    release_dwell_time: float = 0.5
    max_retries: int = 3

    # Downward grasp quaternion (wxyz, world frame).
    # (0, 1, 0, 0) is a 180-deg rotation about X, pointing the Z axis downward.
    # Adjust if the OpenArm EE convention differs.
    grasp_quat: tuple = (0.0, 1.0, 0.0, 0.0)

    # CommandTermCfg: never auto-resample (planner owns its own timing)
    resampling_time_range: tuple = (float("inf"), float("inf"))
    debug_vis: bool = False
