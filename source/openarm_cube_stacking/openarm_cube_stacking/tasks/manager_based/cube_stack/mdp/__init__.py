"""MDP module for the 5-cube stacking task.

Re-exports standard Isaac Lab MDP utilities alongside task-specific functions
so that the environment config can import everything from a single place:

    from . import mdp
    mdp.JointPositionActionCfg(...)
    mdp.reaching_current_cube(...)
"""

# Standard Isaac Lab MDP utilities -------------------------------------------
from isaaclab.envs.mdp import (  # noqa: F401
    action_rate_l2,
    joint_vel_l2,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    reset_root_state_uniform,
    reset_scene_to_default,
    time_out,
)

# Action config classes are in the actions submodule
from isaaclab.envs.mdp.actions import (  # noqa: F401
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)

# Task-specific observation terms --------------------------------------------
from .observations import (  # noqa: F401
    compute_current_cube_idx,
    cube_orientations,
    cube_positions_in_env_frame,
    current_cube_index_obs,
    current_cube_to_target,
    current_target_position_in_env_frame,
    ee_position_in_env_frame,
    ee_to_current_cube,
)

# Task-specific reward terms -------------------------------------------------
from .rewards import (  # noqa: F401
    cube_drop_penalty,
    cube_placed_bonus,
    lifting_current_cube,
    moving_current_cube_to_target,
    placing_current_cube,
    reaching_current_cube,
    stack_success_bonus,
)

# Task-specific termination terms --------------------------------------------
from .terminations import (  # noqa: F401
    all_cubes_stacked,
    cube_dropped,
    stack_collapsed,
)

# Task-specific event functions -----------------------------------------------
from .events import (  # noqa: F401
    reset_cubes_non_overlapping,
    reset_robot_to_default,
    reset_stack_target,
)
