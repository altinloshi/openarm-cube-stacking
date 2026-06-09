# Architecture — OpenArm Cube Stacking

**Developed by Nepher Robotics — contact@nepher.ai**

---

## Overview

The OpenArm Cube Stacking project uses a **two-level hierarchical decomposition**
to reliably stack five DexCube objects in Isaac Lab simulation.

```
┌─────────────────────────────────────────────────────┐
│                High-Level (HL) Layer                │
│          ClassicalStackPlanner (state machine)      │
│     Reads live cube positions → emits EE waypoints  │
└────────────────────┬────────────────────────────────┘
                     │ waypoints: (x, y, z, quat, grip)
                     ▼
┌─────────────────────────────────────────────────────┐
│                Low-Level (LL) Layer                 │
│       Goal-conditioned EE tracking policy           │
│           Trained via PPO (RSL-RL)                  │
└────────────────────┬────────────────────────────────┘
                     │ joint position targets
                     ▼
┌─────────────────────────────────────────────────────┐
│          DifferentialIK + Binary Gripper            │
│                  OpenArm (6-DOF)                    │
└─────────────────────────────────────────────────────┘
```

---

## Low-Level Policy

The LL policy is a **goal-conditioned end-effector tracker** trained in isolation.
It receives a target EE pose and gripper command, and outputs joint position targets.

- **Observation**: joint positions, joint velocities, current EE pose, target EE pose,
  grip command, gripper opening fraction, last action.
- **Action**: 6D IK-relative delta `(Δx, Δy, Δz, Δrx, Δry, Δrz)` + 1D binary gripper.
- **Reward**: weighted combination of EE position error (L2 + tanh), orientation error,
  gripper tracking, action smoothness, and joint velocity penalties.
- **Network**: MLP [256, 128, 64] with ELU activations.
- **Curriculum**: action-rate / joint-velocity penalties ramped during training.

Training environment: `Nepher-OpenArm-CubeStack-LL-v0` (random EE pose commands, no cubes).

---

## High-Level Classical Planner

The HL layer is a **pure-Python / PyTorch state machine** (`ClassicalStackPlanner`) that
operates over a batch of environments simultaneously.

### Planner Stages

For each of the 5 cubes (processed sequentially per environment):

| Stage | Description |
|---|---|
| `PRE_GRASP` | Move EE above the current cube at a safe height |
| `DESCEND` | Lower EE to the cube grasp pose |
| `GRASP` | Close gripper; dwell to ensure contact |
| `LIFT` | Raise EE + cube to transport height |
| `MOVE_ABOVE_STACK` | Translate above the target stack XY position |
| `LOWER_TO_STACK` | Lower to the target height for cube i on the stack |
| `RELEASE` | Open gripper; dwell |
| `RETRACT` | Raise EE clear of the stack |
| `NEXT_CUBE` | Advance cube index; loop back to PRE_GRASP |
| `DONE` | All 5 cubes placed; hold position |

### Stage Transition Criteria

A stage advances when:
1. EE position error < `pos_tolerance` (metres)
2. EE orientation error < `ori_tolerance` (radians)
3. `dwell_time` seconds have elapsed since the goal was reached

### Scene Layout

All HL/Eval environments use the official OpenArm lift-cube scene geometry:
- `SeattleLabTable` workbench
- OpenArm at the official lift-task base pose
- 5 × DexCube objects (scale 0.8) near the official lift-cube spawn strip
- Stack target at a fixed XY position in the reachable workspace

---

## Package Layout

```
source/openarm_cube_stacking/
├── config/
│   └── extension.toml          ← Isaac Lab extension metadata
├── setup.py
└── openarm_cube_stacking/
    ├── __init__.py
    └── tasks/
        └── manager_based/
            └── cube_stack/
                ├── ll_policy/              ← LL training environment
                │   ├── ll_env_cfg.py
                │   ├── ll_env_cfg_play.py
                │   ├── agents/
                │   └── mdp/
                │       ├── commands.py     ← UniformPoseCommandCfg
                │       ├── observations.py ← joint state, EE pose, command
                │       ├── rewards.py      ← EE + gripper tracking
                │       ├── events.py       ← robot + gripper-cmd reset
                │       └── terminations.py ← timeout only
                ├── hl_policy/              ← HL planner + play/eval envs
                │   ├── classical_stack_planner.py  ← core planner
                │   ├── hl_env_cfg.py
                │   ├── hl_env_cfg_play.py
                │   ├── hl_env_cfg_eval.py
                │   ├── agents/
                │   └── mdp/
                │       ├── commands.py     ← ClassicalStackPlannerCommandCfg
                │       ├── observations.py ← LL-compatible + cube state
                │       ├── rewards.py      ← stack progress monitoring
                │       ├── events.py       ← robot + cube + stack resets
                │       └── terminations.py ← timeout + planner done
                ├── eval/                   ← deterministic tournament
                │   ├── scenarios.py        ← 30 fixed scenarios
                │   └── metrics.py          ← StackMetricsAccumulator
                ├── end_to_end/             ← end-to-end baseline envs
                ├── mdp/                    ← shared MDP helpers
                ├── tabletop_scene_cfg.py   ← scene definition
                └── openarm_lift_style_scene_cfg.py  ← compat alias
```

---

## Data Flow (HL Inference)

```
Isaac Lab step()
    │
    ├─ ClassicalStackPlannerCommand.compute()
    │       reads: cube_i.data.root_pos_w, robot EE pose
    │       writes: env.command_manager["ee_pose"] = next_waypoint
    │
    ├─ LL policy.act(obs)
    │       obs = [joint_pos, joint_vel, ee_pos, ee_quat,
    │              target_pos, target_quat, grip_cmd,
    │              gripper_opening, last_action]
    │       action = [Δx, Δy, Δz, Δrx, Δry, Δrz, grip]
    │
    └─ DifferentialIK + BinaryGripper
            → joint position targets → physics step
```
