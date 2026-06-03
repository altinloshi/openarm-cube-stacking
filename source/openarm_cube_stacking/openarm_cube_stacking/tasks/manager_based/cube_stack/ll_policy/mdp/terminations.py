"""Termination terms for the LL policy environment.

The LL task terminates only on timeout; there are no success/failure
conditions tied to cube placement (that is the HL task's concern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate each environment after the episode time limit."""
    return env.episode_length_buf >= env.max_episode_length
