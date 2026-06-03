"""Cube stacking task package.

Registers only the four custom Nepher OpenArm CubeStack environments:
  - Low-level EE tracker        : Nepher-OpenArm-CubeStack-LL-v0 / -LL-Play-v0
  - High-level classical planner: Nepher-OpenArm-CubeStack-HL-Classical-Play-v0
  - Deterministic evaluation    : Nepher-OpenArm-CubeStack-Eval-v0

The LL/HL/Eval environments all use the official OpenArm lift-cube scene setup
(SeattleLabTable + OpenArm mounting) shared via ``tabletop_scene_cfg``.

The previously registered end-to-end task IDs
(``Nepher-OpenArm-CubeStack-v0``, ``-Play-v0``, ``-EndToEnd-v0`` and
``-EndToEnd-Play-v0``) are intentionally no longer registered.  Their config
modules remain on disk for reference but are not exposed as gym environments.
"""

from . import agents  # noqa: F401

# Each sub-package __init__.py registers its own environments via gym.register.
from . import ll_policy  # noqa: F401, E402  (registers LL-v0 / LL-Play-v0)
from . import hl_policy  # noqa: F401, E402  (registers HL-Classical-Play-v0 / Eval-v0)
