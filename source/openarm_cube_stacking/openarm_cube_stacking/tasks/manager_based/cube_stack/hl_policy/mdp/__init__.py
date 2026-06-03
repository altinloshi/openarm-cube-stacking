"""MDP components for the HL classical cube-stacking environment.

Pulls in the full standard Isaac Lab MDP library plus the LL custom MDP terms
(so the HL observation layout is identical to LL training), then re-exports the
HL-specific command / observation / event / reward / termination terms.
"""

# Full standard Isaac Lab MDP + reach rewards + LL custom MDP terms.
from ...ll_policy.mdp import *  # noqa: F401, F403

# HL-specific command terms.
from .commands import (  # noqa: F401
    HLGripCommand,
    HLGripCommandCfg,
    HLStackPoseCommand,
    HLStackPoseCommandCfg,
    make_target_marker_cfg,
)

# HL-specific observation terms.
from .observations import current_cube_index, planner_stage  # noqa: F401

# HL-specific reset events.
from .events import (  # noqa: F401
    reset_cubes_and_stack,
    reset_from_scenarios,
    reset_robot_to_default,
)

# HL-specific rewards.
from .rewards import stack_progress  # noqa: F401

# HL-specific terminations.
from .terminations import all_cubes_stacked, cube_stacked_mask, stack_failed  # noqa: F401
