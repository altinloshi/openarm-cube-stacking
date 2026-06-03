"""Backward-compatibility shim for the old tabletop scene configuration.

All authoritative definitions now live in ``openarm_lift_style_scene_cfg``.
This module re-exports them under the old tabletop-style names so that
existing imports in sub-packages continue to work without modification.

New code should import directly from ``openarm_lift_style_scene_cfg``.
"""

from __future__ import annotations

# Re-export everything from the canonical module.
from .openarm_lift_style_scene_cfg import (  # noqa: F401
    CUBE_GROUND_Z,
    CUBE_HEIGHT,
    CUBE_NAMES,
    CUBE_SIZE,
    LIFT_STYLE_CUBE_SPAWN_POSITIONS,
    LIFT_STYLE_STACK_BASE_LOCAL_POS,
    NUM_CUBES,
    PLANNER_FLOOR_Z,
    TABLE_TOP_Z,
    OpenArmLiftStyleSceneCfg,
    OpenArmLiftStyleWithCubesSceneCfg,
    _EE_MARKER_CFG,
    _EE_TARGET_MARKER_CFG,
    make_dex_cube_cfg,
)

# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility aliases
# ─────────────────────────────────────────────────────────────────────────────

# Old table geometry constants (robot is now floor-mounted, no table exists).
TABLE_HEIGHT: float = 0.0
TABLE_CENTER_X: float = 0.0
TABLE_CENTER_Y: float = 0.0

# Old "cube on table" z  →  cube on floor z
CUBE_TABLE_Z: float = CUBE_GROUND_Z  # 0.055

# Old robot-on-table placement constants  →  robot is at env origin
ROBOT_ON_TABLE_X: float = 0.0
ROBOT_ON_TABLE_Y: float = 0.0
ROBOT_ON_TABLE_Z: float = 0.0
OPENARM_BASE_Z_OFFSET: float = 0.0

# Old spawn / stack position names
TABLETOP_CUBE_SPAWN_LOCAL_POSITIONS = LIFT_STYLE_CUBE_SPAWN_POSITIONS
TABLETOP_STACK_BASE_LOCAL_POS = LIFT_STYLE_STACK_BASE_LOCAL_POS

# Old scene class names
OpenArmTabletopSceneCfg = OpenArmLiftStyleSceneCfg
OpenArmTabletopWithCubesSceneCfg = OpenArmLiftStyleWithCubesSceneCfg
