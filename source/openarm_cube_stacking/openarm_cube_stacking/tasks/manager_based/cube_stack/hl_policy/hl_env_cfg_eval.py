"""Evaluation environment config for Nepher-OpenArm-CubeStack-Eval-v0.

Deterministic, reproducible tournament scenarios:
- 30 predefined scenarios (cube positions, stack target)
- scenario_idx = env_id % 30
- no observation noise
- no random spawn
- fixed seed
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..eval.scenarios import (
    EVAL_SCENARIOS,
    reset_cubes_from_scenario,
    reset_stack_from_scenario,
)
from ..openarm_lift_style_scene_cfg import CUBE_NAMES
from . import mdp
from .hl_env_cfg import HLSceneCfg, OpenArmHLEnvCfg


@configclass
class EvalSceneCfg(HLSceneCfg):
    """Eval scene: debug vis off, deterministic."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.ee_frame.debug_vis = False


@configclass
class EvalEventCfg:
    """Deterministic reset events using scenario lookup table."""

    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reset_cubes_eval = EventTerm(
        func=reset_cubes_from_scenario,
        mode="reset",
        params={
            "cube_names": list(CUBE_NAMES),
            "scenarios": EVAL_SCENARIOS,
        },
    )
    reset_stack_eval = EventTerm(
        func=reset_stack_from_scenario,
        mode="reset",
        params={"scenarios": EVAL_SCENARIOS},
    )


@configclass
class OpenArmEvalEnvCfg(OpenArmHLEnvCfg):
    """Deterministic evaluation environment for tournament scoring."""

    scene: EvalSceneCfg = EvalSceneCfg(num_envs=30, env_spacing=2.5)
    events: EvalEventCfg = EvalEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 30
        self.scene.env_spacing = 2.5
        # Deterministic: no noise in observations
        self.observations.policy.enable_corruption = False
        self.observations.stack_state.enable_corruption = False
        # Planner command: no debug vis during eval
        self.commands.ee_pose.debug_vis = False
        # Episode length long enough for full stack
        self.episode_length_s = 120.0
