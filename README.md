# OpenArm Cube Stacking — 5-Cube Hierarchical Stacking

**Developed by Nepher Robotics — contact@nepher.ai**

Isaac Lab external project for hierarchical OpenArm 5-cube stacking.

## Architecture

```
HL Classical Planner  — ClassicalStackPlanner state machine (5 cubes)
    ↓  waypoints (x, y, z, quat, grip)
LL Policy  — goal-conditioned EE tracker  [Phase 1 — trained]
    ↓  target (x, y, z, rx, ry, rz, grip)
Joint-position control + binary gripper
    ↓
OpenArm (6-DOF + gripper)
```

The system uses a **two-level hierarchical decomposition**:

- The **Low-Level (LL) policy** is a goal-conditioned end-effector tracker trained
  with PPO (RSL-RL).  It takes a commanded EE pose and gripper command, and
  outputs joint position targets.
- The **High-Level (HL) classical planner** is a pure-PyTorch state machine that
  reads live cube positions and emits a sequence of EE target poses to pick up
  each cube and place it on the stack.

## Environment Overview

| Environment ID | Description |
|---|---|
| `Nepher-OpenArm-CubeStack-LL-v0` | Low-level EE tracker — training |
| `Nepher-OpenArm-CubeStack-LL-Play-v0` | Low-level EE tracker — evaluation |
| `Nepher-OpenArm-CubeStack-HL-Classical-Play-v0` | Classical planner + frozen LL policy |
| `Nepher-OpenArm-CubeStack-Eval-v0` | Deterministic 30-scenario tournament evaluation |

## Low-Level Policy

| | |
|---|---|
| **Action** | 6D IK-Rel `(Δx,Δy,Δz,Δrx,Δry,Δrz)` + 1D binary gripper |
| **Observation** | joint pos/vel, EE pose, target EE pose, grip cmd, gripper pos, last action |
| **Command** | `ee_pose` (resampled each episode) + `grip_cmd` (per-episode) |
| **Reward** | Coarse L2 + fine tanh for position & orientation; soft gripper match; smoothness |
| **Network** | MLP [256, 128, 64], ELU, PPO (RSL-RL) |

## High-Level Classical Stack Planner

10-stage `ClassicalStackPlanner` drives the HL commands; the frozen LL policy executes them.

| | |
|---|---|
| **Stages** | PRE_GRASP → DESCEND → GRASP → LIFT → MOVE_ABOVE_STACK → LOWER_TO_STACK → RELEASE → RETRACT → NEXT_CUBE → DONE |
| **Transition** | EE error < tolerance + dwell time |
| **Cubes** | 5 × DexCube (scale 0.8), random spawn near official OpenArm lift-cube strip |
| **Goal** | Fixed stack target at `[0.55, 0.0, 0.055]` |
| **Final method** | Classical/manual motion planning (state machine) — the reliable, evaluated pipeline |

> **Note on learning-based HL approaches:** Robomimic and RL-based HL policies were
> explored experimentally during development. The classical state-machine planner is
> the final selected and evaluated approach.  No high success-rate figures from
> learning-based methods are claimed.

## Install

```bash
python -m pip install -e source/openarm_cube_stacking
```

## Verify Environment Registration

```bash
python scripts/list_envs.py
```

## Train

```bash
# Train LL EE-tracking policy (headless recommended)
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-LL-v0 --headless --num_envs=4096

# Resume training
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-LL-v0 --headless --resume
```

Checkpoints → `logs/rsl_rl/openarm_ll_ee_tracking/<timestamp>/`.

## Export Policy

```bash
python scripts/rsl_rl/export_policy.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

Exports to:
```
best_policy/best_policy.pt
best_policy/exported/policy.pt    ← TorchScript
best_policy/exported/policy.onnx  ← ONNX (opset 17)
```

## Evaluate (LL)

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

## Run Classical Stack Planner (HL)

```bash
python scripts/rsl_rl/play.py \
    --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 \
    --checkpoint=best_policy/best_policy.pt \
    --video --video_length=600
```

## Deterministic Tournament Evaluation (30 Scenarios)

```bash
python scripts/eval/evaluate_stack.py \
    --task=Nepher-OpenArm-CubeStack-Eval-v0 \
    --num_envs=30 \
    --episodes=30
```

Results → `logs/eval/openarm_cube_stack/<timestamp>/results.json`.

```bash
# Generate human-readable report
python scripts/eval/make_eval_report.py logs/eval/openarm_cube_stack/
```

---

## Reproducing Tournament Scoring

```bash
# 1. Train LL to convergence
python scripts/rsl_rl/train.py \
    --task=Nepher-OpenArm-CubeStack-LL-v0 \
    --headless --num_envs=4096 --max_iterations=3000

# 2. Export best checkpoint
python scripts/rsl_rl/export_policy.py \
    --task=Nepher-OpenArm-CubeStack-LL-Play-v0

# 3. Run tournament
python scripts/eval/evaluate_stack.py \
    --task=Nepher-OpenArm-CubeStack-Eval-v0 \
    --num_envs=30 --episodes=30

# 4. Report
python scripts/eval/make_eval_report.py logs/eval/openarm_cube_stack/
```

---

## Utility Scripts

```bash
# List all registered environments
python scripts/list_envs.py

# Random actions in LL env (quick scene check)
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-LL-v0 --num_envs=2

# Zero actions in HL play env
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --num_envs=2
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `full_stack_success_rate` | Fraction of episodes with all 5 cubes stacked and stable |
| `average_cubes_stacked` | Mean cubes placed per episode |
| `per_cube_success_rate` | Per-cube stacking success rate [cube 1..5] |
| `cube_drop_rate` | Fraction of episodes with at least one cube dropped |
| `stack_collapse_rate` | Fraction of episodes with stack collapse after placement |
| `mean_final_stack_error` | Mean positional error to target at episode end (m) |
| `mean_episode_length_s` | Mean episode duration (s) |
| `mean_grasp_retries` | Mean grasp retries per episode |
| `timeout_rate` | Fraction of episodes that hit the time limit |

A cube is **successfully stacked** when:
- XY error to target < 0.04 m
- Height error to expected stack level < 0.02 m
- Linear velocity < 0.05 m/s
- All cubes below it remain stable

---

## Repository Structure

```
.
├── docs/
│   ├── assets/            # diagrams and images
│   ├── architecture.md    # system architecture overview
│   └── algorithm.md       # final algorithm description
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   ├── rsl_rl/
│   │   ├── train.py
│   │   ├── play.py
│   │   ├── cli_args.py
│   │   └── export_policy.py
│   └── eval/
│       ├── evaluate_stack.py
│       └── make_eval_report.py
├── source/
│   └── openarm_cube_stacking/
│       ├── config/
│       │   └── extension.toml
│       ├── setup.py
│       └── openarm_cube_stacking/
│           ├── tasks/
│           │   └── manager_based/
│           │       └── cube_stack/
│           │           ├── ll_policy/     ← LL EE-tracker env + MDP
│           │           ├── hl_policy/     ← classical planner + HL env
│           │           ├── end_to_end/    ← end-to-end baseline envs
│           │           ├── eval/          ← tournament evaluation framework
│           │           └── mdp/           ← shared MDP helpers
│           └── __init__.py
├── logs/          ← ignored (training outputs)
├── datasets/      ← ignored (recorded datasets)
├── best_policy/   ← ignored (exported policy artifacts)
├── LICENSE
├── .gitignore
└── README.md
```

---

## Requirements

- Isaac Lab (with `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl` namespaces)
- Isaac Sim compatible with your Isaac Lab version
- Python 3.10+

---

## Known Issues / Limitations

- The LL policy must be fully trained before the HL classical planner can be used.
- Robomimic / imitation learning–based pipelines were explored experimentally; they
  are **not** the final selected approach and are not documented as a reliable method.
- No trained checkpoints are committed to this repository. You must train from scratch.
- Evaluation results in the repo template show placeholder zero values; real results
  depend on a fully trained LL policy.

---

## License

Copyright (c) 2025-2026, Nepher Robotics.
Licensed under the BSD-3-Clause License. See `LICENSE` for details.
