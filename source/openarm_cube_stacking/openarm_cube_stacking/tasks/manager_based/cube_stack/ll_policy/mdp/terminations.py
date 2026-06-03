from __future__ import annotations

"""Termination terms for the OpenArm LL policy.

The LL EE-tracking task terminates on timeout only (it is a continuous tracking
task with no success/failure event). The standard ``time_out`` termination is
re-exported through this package's ``mdp`` namespace; this module exists to keep
the LL MDP layout symmetric with the HL package and to host any future
LL-specific terminations.
"""
