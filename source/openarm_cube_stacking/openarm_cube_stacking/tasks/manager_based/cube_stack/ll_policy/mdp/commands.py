from __future__ import annotations

"""Custom command terms for the OpenArm low-level (LL) policy environment.

GripperCommand / GripperCommandCfg
----------------------------------
Samples a binary gripper target (0 = open, 1 = close) once per episode at
environment reset. The resampling timer is set far beyond any episode length so
it never fires mid-episode; resampling happens only via the reset path in the
command manager. ``close_prob`` controls the open/close duty cycle (default 0.5
=> 50 % open, 50 % close episodes).
"""

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GripperCommand(CommandTerm):
    """Binary gripper target resampled once per episode.

    Output tensor shape ``(num_envs, 1)``: ``0.0`` (open) or ``1.0`` (close).
    """

    cfg: GripperCommandCfg

    def __init__(self, cfg: GripperCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._grip_target: torch.Tensor = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "GripperCommand\n"
        msg += f"\tCommand dimension: {self.command_dim}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self._grip_target

    @property
    def command_dim(self) -> int:
        return 1

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        self._grip_target[env_ids, 0] = (torch.rand(n, device=self.device) < self.cfg.close_prob).float()

    def _update_command(self) -> None:
        # Static within an episode.
        pass

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class GripperCommandCfg(CommandTermCfg):
    """Configuration for :class:`GripperCommand`."""

    class_type: type = GripperCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    # Probability of commanding a *close* at episode reset (0.5 => balanced).
    close_prob: float = 0.5
