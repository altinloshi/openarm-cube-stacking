from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GripperCommand(CommandTerm):
    """Binary gripper target, resampled at episode reset.

    Command convention: 0.0 = open, 1.0 = close.
    """

    cfg: "GripperCommandCfg"

    def __init__(self, cfg: "GripperCommandCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self._grip_target = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._grip_target

    @property
    def command_dim(self) -> int:
        return 1

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._grip_target[env_ids, 0] = (
            torch.rand(env_ids.shape[0], device=self.device) < self.cfg.close_prob
        ).float()

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class GripperCommandCfg(CommandTermCfg):
    """Configuration for the reset-sampled binary gripper command."""

    class_type: type = GripperCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    close_prob: float = 0.5
