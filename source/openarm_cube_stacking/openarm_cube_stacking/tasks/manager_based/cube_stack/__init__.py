# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nepher OpenArm CubeStack task registrations.

Only the four official custom CubeStack task IDs are registered here:

- Nepher-OpenArm-CubeStack-LL-v0
- Nepher-OpenArm-CubeStack-LL-Play-v0
- Nepher-OpenArm-CubeStack-HL-Classical-Play-v0
- Nepher-OpenArm-CubeStack-Eval-v0

The old baseline / EndToEnd task IDs are intentionally not registered.
"""

# Import subpackages that perform the actual gym.register calls.
# Do not import the old baseline or end_to_end modules here.
from . import ll_policy  # noqa: F401
from . import hl_policy  # noqa: F401
