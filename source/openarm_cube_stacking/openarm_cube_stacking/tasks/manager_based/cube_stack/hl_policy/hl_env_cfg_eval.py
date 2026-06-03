from __future__ import annotations

"""Deterministic evaluation configuration for HL classical cube stacking.

Used by ``Nepher-OpenArm-CubeStack-Eval-v0``. Differences from the HL play env:

* Deterministic 30-scenario reset (``reset_from_scenarios``); scenario index is
  ``env_id % 30``. No random cube spawn, no random stack base.
* No observation noise.
* Planner logging disabled (keeps the eval console clean).

Run via the evaluation script::

    python scripts/eval/evaluate_stack.py --task=Nepher-OpenArm-CubeStack-Eval-v0 \
        --num_envs=30 --episodes=30
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import tabletop
from . import mdp
from .hl_env_cfg import HLEnvCfg


@configclass
class HLEvalEventCfg:
    """Deterministic reset events from the pre-baked tournament scenarios."""

    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_from_scenarios = EventTerm(
        func=mdp.reset_from_scenarios,
        mode="reset",
        params={
            "cube_names": list(tabletop.CUBE_NAMES),
            "pose_cmd_name": "ee_pose",
        },
    )


@configclass
class HLEnvCfg_EVAL(HLEnvCfg):
    """Deterministic tournament-evaluation configuration (30 scenarios)."""

    events: HLEvalEventCfg = HLEvalEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # One environment per deterministic scenario by default.
        self.scene.num_envs = 30
        self.scene.env_spacing = 2.5
        # No noise, no logging, reproducible.
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.enable_log = False
        self.commands.ee_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False
        self.seed = 0
