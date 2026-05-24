"""Play (evaluation) variant of the 5-cube stacking environment.

Reduces the environment count and disables observation noise so a trained
policy can be evaluated visually.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .cube_stack_env_cfg import OpenArmCubeStackEnvCfg


@configclass
class OpenArmCubeStackEnvCfg_PLAY(OpenArmCubeStackEnvCfg):
    """Smaller, noiseless environment for visual playback."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Fewer envs so they fit on screen
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation corruption (noise) during evaluation
        self.observations.policy.enable_corruption = False

        # Randomisation is turned off for deterministic playback
        self.events.reset_cube_0 = None  # type: ignore[assignment]
        self.events.reset_cube_1 = None  # type: ignore[assignment]
        self.events.reset_cube_2 = None  # type: ignore[assignment]
        self.events.reset_cube_3 = None  # type: ignore[assignment]
        self.events.reset_cube_4 = None  # type: ignore[assignment]
