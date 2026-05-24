"""List Gym environments registered by ``openarm_cube_stacking``.

Usage::

    python scripts/list_envs.py
"""

from __future__ import annotations

import gymnasium as gym

# Side-effect import: registers the Nepher-OpenArm-CubeStack-* envs.
import openarm_cube_stacking.tasks  # noqa: F401


KEYWORDS = ("OpenArm", "CubeStack", "Nepher")


def main() -> None:
    matched = [eid for eid in gym.registry.keys() if any(k in eid for k in KEYWORDS)]
    matched.sort()
    if not matched:
        print("No OpenArm / CubeStack / Nepher environments found.")
        print("Did you `pip install -e source/openarm_cube_stacking`?")
        return
    print("Registered OpenArm cube-stacking environments:")
    for eid in matched:
        print(f"  - {eid}")


if __name__ == "__main__":
    main()
