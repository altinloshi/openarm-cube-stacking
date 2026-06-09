# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end environment config for Nepher-OpenArm-CubeStack-EndToEnd-v0.

This is the same lift-style baseline as the top-level cube_stack task, exposed
under the end_to_end sub-package name.
"""

from isaaclab.utils import configclass

from ..cube_stack_env_cfg import OpenArmCubeStackEnvCfg


@configclass
class OpenArmCubeStackEndToEndEnvCfg(OpenArmCubeStackEnvCfg):
    """Alias for the end-to-end baseline cube stacking environment."""
    pass
