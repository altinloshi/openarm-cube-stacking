from __future__ import annotations

"""Tournament metrics for OpenArm cube stacking.

Pure-torch helpers (no Isaac Lab dependency) plus a :class:`StackEvaluator`
accumulator that aggregates per-episode results into the final tournament report.

Success definition (per cube)
-----------------------------
A cube counts as successfully stacked when:
  * its centre is within ``xy_tol`` of the target slot centre (horizontal),
  * its height is within ``z_tol`` of the expected stack height,
  * it is not moving above ``vel_tol`` (stable),
  * (implicitly) the cubes below it are also stacked — enforced by requiring the
    whole prefix of the stack to be placed.

Full-stack success requires all cubes stacked and stable.
"""

from dataclasses import dataclass, field

import torch


def cubes_stacked_mask(
    cube_pos: torch.Tensor,   # (N, M, 3)
    cube_vel: torch.Tensor,   # (N, M, 3)
    target_pos: torch.Tensor,  # (N, M, 3)
    xy_tol: float = 0.03,
    z_tol: float = 0.025,
    vel_tol: float = 0.05,
) -> torch.Tensor:
    """Per-cube boolean mask: placed, at the right height, and stable."""
    xy_err = torch.norm(cube_pos[..., :2] - target_pos[..., :2], dim=-1)
    z_err = (cube_pos[..., 2] - target_pos[..., 2]).abs()
    speed = torch.norm(cube_vel, dim=-1)
    return (xy_err < xy_tol) & (z_err < z_tol) & (speed < vel_tol)


def stacked_prefix_mask(stacked: torch.Tensor) -> torch.Tensor:
    """Keep only the contiguous stacked prefix (a cube counts only if every cube
    below it is also stacked). ``stacked`` is ``(N, M)`` boolean ordered bottom→top.
    """
    # cumulative-AND along the cube axis.
    return torch.cumprod(stacked.to(torch.int64), dim=1).to(torch.bool)


def final_stack_error(
    cube_pos: torch.Tensor,    # (N, M, 3)
    target_pos: torch.Tensor,  # (N, M, 3)
) -> torch.Tensor:
    """Mean per-env Euclidean error of all cubes to their target slots ``(N,)``."""
    return torch.norm(cube_pos - target_pos, dim=-1).mean(dim=1)


def cubes_dropped_mask(cube_pos: torch.Tensor, min_height: float) -> torch.Tensor:
    """``(N,)`` bool: any cube fell below ``min_height``."""
    return (cube_pos[:, :, 2] < min_height).any(dim=1)


@dataclass
class StackEvaluator:
    """Accumulate per-episode stacking results and compute tournament metrics."""

    num_cubes: int = 5

    # Per-episode records (one entry per completed episode).
    _per_cube_stacked: list[torch.Tensor] = field(default_factory=list)  # each (M,)
    _full_success: list[bool] = field(default_factory=list)
    _dropped: list[bool] = field(default_factory=list)
    _collapsed: list[bool] = field(default_factory=list)
    _stack_error: list[float] = field(default_factory=list)
    _episode_len_s: list[float] = field(default_factory=list)
    _grasp_retries: list[float] = field(default_factory=list)
    _timeout: list[bool] = field(default_factory=list)

    def record_episode(
        self,
        per_cube_stacked: torch.Tensor,  # (M,) bool — contiguous stacked prefix
        dropped: bool,
        collapsed: bool,
        stack_error: float,
        episode_len_s: float,
        grasp_retries: float,
        timeout: bool,
    ) -> None:
        stacked = per_cube_stacked.detach().cpu().bool()
        self._per_cube_stacked.append(stacked)
        self._full_success.append(bool(stacked.all().item()))
        self._dropped.append(bool(dropped))
        self._collapsed.append(bool(collapsed))
        self._stack_error.append(float(stack_error))
        self._episode_len_s.append(float(episode_len_s))
        self._grasp_retries.append(float(grasp_retries))
        self._timeout.append(bool(timeout))

    @property
    def num_episodes(self) -> int:
        return len(self._full_success)

    def compute(self) -> dict:
        """Return the tournament metrics as a JSON-serialisable dict."""
        n = self.num_episodes
        if n == 0:
            return self._empty()

        stacked = torch.stack(self._per_cube_stacked, dim=0).float()  # (E, M)
        per_cube = stacked.mean(dim=0).tolist()  # (M,)
        cubes_per_ep = stacked.sum(dim=1)  # (E,)

        def _rate(flags: list[bool]) -> float:
            return float(sum(1 for f in flags if f) / n)

        def _mean(vals: list[float]) -> float:
            return float(sum(vals) / n)

        return {
            "full_stack_success_rate": _rate(self._full_success),
            "average_cubes_stacked": float(cubes_per_ep.mean().item()),
            "per_cube_success_rate": [round(p, 6) for p in per_cube],
            "cube_drop_rate": _rate(self._dropped),
            "stack_collapse_rate": _rate(self._collapsed),
            "mean_final_stack_error": _mean(self._stack_error),
            "mean_episode_length_s": _mean(self._episode_len_s),
            "mean_grasp_retries": _mean(self._grasp_retries),
            "timeout_rate": _rate(self._timeout),
        }

    def _empty(self) -> dict:
        return {
            "full_stack_success_rate": 0.0,
            "average_cubes_stacked": 0.0,
            "per_cube_success_rate": [0.0] * self.num_cubes,
            "cube_drop_rate": 0.0,
            "stack_collapse_rate": 0.0,
            "mean_final_stack_error": 0.0,
            "mean_episode_length_s": 0.0,
            "mean_grasp_retries": 0.0,
            "timeout_rate": 0.0,
        }
