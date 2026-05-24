"""List all registered Gym environments from the openarm_cube_stacking package.

Usage::

    python scripts/list_envs.py

The script does NOT launch Isaac Sim — it only imports the package so that
gymnasium.register() calls are executed, then prints all environments whose
IDs contain "OpenArm", "CubeStack", or "Nepher".
"""

from __future__ import annotations

import gymnasium as gym

# Trigger gym.register() calls inside the package
import openarm_cube_stacking.tasks  # noqa: F401

KEYWORDS = ("OpenArm", "CubeStack", "Nepher")


def main() -> None:
    print("\nRegistered environments matching keywords:", KEYWORDS)
    print("-" * 60)

    found: list[str] = []
    for env_id in gym.envs.registry:
        if any(kw in env_id for kw in KEYWORDS):
            found.append(env_id)

    if not found:
        print("  (none found – check that the package is installed)")
    else:
        for env_id in sorted(found):
            spec = gym.spec(env_id)
            print(f"  {env_id}")
            kw = spec.kwargs or {}
            if "env_cfg_entry_point" in kw:
                print(f"    env_cfg   : {kw['env_cfg_entry_point']}")
            if "rsl_rl_cfg_entry_point" in kw:
                print(f"    rsl_rl_cfg: {kw['rsl_rl_cfg_entry_point']}")

    print(f"\nTotal: {len(found)} environment(s)")


if __name__ == "__main__":
    main()
