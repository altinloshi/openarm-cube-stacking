from __future__ import annotations

"""Custom reset events for the OpenArm LL policy.

The LL EE-tracking environment relies primarily on the standard Isaac Lab
``reset_joints_by_scale`` event (re-exported through this package's ``mdp``
namespace) to randomise the arm start configuration each episode. This module
hosts any LL-specific reset helpers; it is intentionally lightweight because the
LL policy does not interact with cubes.
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_gripper_to_open(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
    open_val: float = 0.044,
) -> None:
    """Reset the gripper fingers to the fully-open position on episode reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = robot.data.joint_pos[env_ids][:, joint_ids].clone()
    joint_pos[:] = open_val
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
