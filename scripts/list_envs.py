"""List registered gym environments relevant to the OpenArm cube stack task."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "openarm_cube_stacking"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import openarm_cube_stacking.tasks  # noqa: F401,E402


def main() -> None:
    matching_envs = sorted(
        env_id
        for env_id in gym.registry.keys()
        if any(token in env_id for token in ("OpenArm", "CubeStack", "Nepher"))
    )
    if not matching_envs:
        print("No matching environments found.")
        return

    print("Registered environments:")
    for env_id in matching_envs:
        print(f" - {env_id}")


if __name__ == "__main__":
    main()
