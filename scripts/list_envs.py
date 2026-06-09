#!/usr/bin/env python3
# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import openarm_cube_stacking.tasks  # noqa: F401, E402


def main():
    print("\nRegistered OpenArm / CubeStack environments:")
    for env_id in sorted(gym.registry.keys()):
        if "OpenArm" in env_id or "CubeStack" in env_id or "Nepher" in env_id:
            print(f"  {env_id}")


if __name__ == "__main__":
    main()
    simulation_app.close()
