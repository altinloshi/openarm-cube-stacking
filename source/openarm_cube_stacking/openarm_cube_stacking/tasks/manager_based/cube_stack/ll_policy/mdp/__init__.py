"""MDP components for the OpenArm LL goal-conditioned EE-tracking environment.

Re-exports the full standard Isaac Lab MDP library plus the reach-task tracking
reward helpers, then adds the LL-specific command / observation / reward terms
so ``ll_env_cfg.py`` can import everything through this single ``mdp`` namespace.
"""

# Full standard Isaac Lab MDP library (observations, rewards, terminations,
# events, commands, curriculum utilities ...).
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Standard reach-task reward functions (coarse L2 + fine tanh position tracking).
from isaaclab_tasks.manager_based.manipulation.reach.mdp import (  # noqa: F401
    orientation_command_error,
    position_command_error,
    position_command_error_tanh,
)

# Custom command terms.
from .commands import GripperCommand, GripperCommandCfg  # noqa: F401

# Custom observation terms.
from .observations import (  # noqa: F401
    OPENARM_GRIPPER_OPEN_VAL,
    ee_pose_in_robot_base,
    grip_command_obs,
    gripper_pos_normalized,
)

# Custom reward terms.
from .rewards import (  # noqa: F401
    gripper_command_tracking,
    orientation_command_error_tanh,
)

# Custom events.
from .events import reset_gripper_to_open  # noqa: F401
