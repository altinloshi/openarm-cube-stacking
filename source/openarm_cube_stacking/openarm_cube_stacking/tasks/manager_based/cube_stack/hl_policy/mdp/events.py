from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from ...end_to_end.mdp.events import reset_cubes_non_overlapping, reset_robot_to_default, reset_stack_target
from ...tabletop_scene import CUBE_NAMES, DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS, DEFAULT_STACK_BASE_LOCAL_POS
from ..mdp.commands import HLPoseCommand
from ...eval.scenarios import scenario_tensors


def reset_hl_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    cube_names: Sequence[str] = CUBE_NAMES,
    local_positions: Sequence[Sequence[float]] = DEFAULT_CUBE_SPAWN_LOCAL_POSITIONS,
    local_stack_base: Sequence[float] = DEFAULT_STACK_BASE_LOCAL_POS,
    cube_position_noise: float = 0.015,
    stack_position_noise: float = 0.02,
    pose_cmd_name: str = "ee_pose",
) -> None:
    """Reset robot, cubes, and stack target for HL play/training-style runs."""
    reset_robot_to_default(env, env_ids, SceneEntityCfg("robot"))
    reset_stack_target(env, env_ids, local_stack_base=local_stack_base, position_noise=stack_position_noise)
    reset_cubes_non_overlapping(
        env, env_ids, cube_names=cube_names, local_positions=local_positions, position_noise=cube_position_noise
    )
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    pose_term.set_stack_base_from_local(ids, env.stack_base_pos_w[ids] - env.scene.env_origins[ids])


def reset_hl_scene_from_scenarios(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_cmd_name: str = "ee_pose",
) -> None:
    """Reset eval envs from deterministic scenario ``env_id % 30``."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
    reset_robot_to_default(env, ids, SceneEntityCfg("robot"))
    cube_pos, cube_quat, stack_base = scenario_tensors(ids, env.device)
    origins = env.scene.env_origins[ids]
    velocity = torch.zeros((ids.numel(), 6), device=env.device)
    for cube_id, cube_name in enumerate(CUBE_NAMES):
        cube: RigidObject = env.scene[cube_name]
        world_pos = origins + cube_pos[:, cube_id]
        cube.write_root_pose_to_sim(torch.cat((world_pos, cube_quat[:, cube_id]), dim=-1), env_ids=ids)
        cube.write_root_velocity_to_sim(velocity, env_ids=ids)
        cube.reset(ids)
    if not hasattr(env, "stack_base_pos_w") or env.stack_base_pos_w.shape[0] != env.num_envs:
        env.stack_base_pos_w = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    env.stack_base_pos_w[ids] = origins + stack_base
    pose_term: HLPoseCommand = env.command_manager.get_term(pose_cmd_name)
    pose_term.set_stack_base_from_local(ids, stack_base)
