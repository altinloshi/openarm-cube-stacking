# OpenArm Cube Stacking for Isaac Lab

Developed by Nepher AI – contact@nepher.ai

## Overview

This repository implements a **hierarchical, tournament-ready cube-stacking pipeline** for the OpenArm robot in Isaac Lab.  The architecture follows a two-level decomposition:

```
HL Policy / Classical Planner
    ↓  waypoints: x, y, z, quat, grip
LL Policy / Goal-conditioned EE tracker
    ↓  tracks target EE pose + gripper command
Joint-position control + binary gripper
    ↓
OpenArm fixed at the standard lift base pose
```

### Visual Setup

All cube-stacking environments reproduce the official
`Isaac-Lift-Cube-OpenArm-Play-v0` scene:
- Multiple parallel Isaac Lab environments
- Standard `SeattleLabTable` lab table (top surface at `z = 0`)
- **OpenArm fixed at the standard lift base pose** (root on the lab-table top,
  not raised onto a custom workbench)
- Five `DexCube` objects (scale 0.8) placed in front of the arm
- Stack target position in the same reachable workspace
- Debug visualisation for the current EE frame, the target EE command frame,
  the stack-target tower, and the current cube index

### Environment Overview

| Environment ID | Description |
|---|---|
| `Nepher-OpenArm-CubeStack-v0` | End-to-end baseline (lift-style scene) |
| `Nepher-OpenArm-CubeStack-Play-v0` | Play variant of the baseline |
| `Nepher-OpenArm-CubeStack-EndToEnd-v0` | EndToEnd alias (same as baseline) |
| `Nepher-OpenArm-CubeStack-EndToEnd-Play-v0` | Play variant alias |
| `Nepher-OpenArm-CubeStack-LL-v0` | **Low-level EE tracker** (lift-style scene, no cubes) |
| `Nepher-OpenArm-CubeStack-LL-Play-v0` | Play variant of LL tracker |
| `Nepher-OpenArm-CubeStack-HL-Classical-Play-v0` | **Classical planner + LL policy** (lift-style scene + cubes) |
| `Nepher-OpenArm-CubeStack-Eval-v0` | **Deterministic 30-scenario tournament evaluation** |

---

## Requirements

- Isaac Lab (with `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl` namespaces)
- Isaac Sim compatible with your Isaac Lab version
- Python 3.10+

## Installation

```bash
python -m pip install -e source/openarm_cube_stacking
```

## Verify Environment Registration

```bash
python scripts/list_envs.py
```

Expected output includes all eight environment IDs listed above.

---

## Project Layout

```
scripts/
  list_envs.py
  random_agent.py
  zero_agent.py
  rsl_rl/
    train.py
    play.py
    cli_args.py
    export_policy.py          ← NEW: exports LL policy to best_policy/
  eval/
    evaluate_stack.py         ← NEW: deterministic 30-scenario evaluation
    make_eval_report.py       ← NEW: human-readable report from results.json

source/openarm_cube_stacking/
  openarm_cube_stacking/
    tasks/manager_based/cube_stack/
      openarm_lift_style_scene_cfg.py  ← shared scene matching Isaac-Lift-Cube-OpenArm
      cube_stack_env_cfg.py   ← existing end-to-end baseline
      cube_stack_env_cfg_play.py
      mdp/                    ← existing MDP modules
      agents/                 ← existing agents
      end_to_end/             ← NEW: EndToEnd environment aliases
        cube_stack_env_cfg.py
        cube_stack_env_cfg_play.py
        mdp/  agents/
      ll_policy/              ← NEW: goal-conditioned EE tracker
        ll_env_cfg.py
        ll_env_cfg_play.py
        mdp/
          commands.py         ← UniformPoseCommandCfg wrapper
          observations.py     ← joint state, EE pose, command targets
          rewards.py          ← EE tracking + gripper rewards
          events.py           ← robot reset + gripper-cmd reset
          terminations.py     ← timeout only
        agents/
          rsl_rl_ppo_cfg.py
      hl_policy/              ← NEW: classical planner + LL execution
        classical_stack_planner.py  ← vectorised torch planner
        hl_env_cfg.py
        hl_env_cfg_play.py
        hl_env_cfg_eval.py
        mdp/
          commands.py         ← ClassicalStackPlannerCommandCfg
          observations.py     ← LL-compatible + cube state
          events.py           ← robot + cube + stack resets
          rewards.py          ← stack progress monitoring
          terminations.py     ← timeout + planner done
        agents/
          rsl_rl_ppo_cfg.py
      eval/                   ← NEW: tournament evaluation framework
        scenarios.py          ← 30 deterministic scenarios
        metrics.py            ← StackMetricsAccumulator
```

---

## 1. End-to-End Baseline (existing task)

The end-to-end five-cube stacking task now uses the same OpenArm lift-style
scene as the hierarchical tasks (its task/MDP behaviour is otherwise unchanged).

```bash
# Train
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless

# Play
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0
```

---

## 2. Hierarchical Pipeline

### Overview

The hierarchical pipeline splits the task into:

- **LL policy** – a fast, goal-conditioned EE tracker trained with RL.
  It takes a commanded EE pose (position + quaternion) and a gripper command,
  and outputs joint position targets.

- **HL classical planner** – a deterministic stage machine that plans cube
  pickup and placement sequences.  It reads cube positions from the scene
  and emits EE target poses consumed by the LL policy.

### OpenArm Lift-Style Scene

All cube-stacking environments reproduce the official
`Isaac-Lift-Cube-OpenArm-Play-v0` scene exactly.  The OpenArm is fixed at the
standard lift base pose (root on the lab-table top), a `SeattleLabTable` USD is
used as the table, and the ground plane is dropped so the table top sits at
`z = 0`.  The robot is **not** raised onto a custom workbench.

```
TABLE_TOP_Z   = 0.0 m            # lab-table top surface (robot root height)
GROUND_Z      = -1.05 m          # ground plane (table top flush with z=0)
CUBE_SCALE    = 0.8              # DexCube scale (matches the lift object)
CUBE_HEIGHT   = 0.064 m          # effective cube edge / stack step
CUBE_SPAWN_Z  = 0.055 m          # cube spawn height (matches the lift object)
```

The single lift object is replaced by five colour-coded `DexCube`s
(`dex_cube_instanceable.usd`) placed in front of the arm:

```
cube_0: [0.35, -0.16, 0.055]
cube_1: [0.35, -0.08, 0.055]
cube_2: [0.40,  0.00, 0.055]
cube_3: [0.35,  0.08, 0.055]
cube_4: [0.35,  0.16, 0.055]
stack base: [0.55, 0.0, 0.055]   # target_z(i) = 0.055 + i * CUBE_HEIGHT
```

Constants and scene classes live in `openarm_lift_style_scene_cfg.py`
(`OpenArmLiftStyleSceneCfg` / `OpenArmLiftStyleWithCubesSceneCfg`).  The EE
frame transformer uses the official OpenArm frame names: root `openarm_link0`,
EE body `openarm_hand`, TCP target `openarm_ee_tcp`.

### Visual verification

Compare the Nepher cube-stack scenes against the official OpenArm lift scene:

```bash
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Isaac-Lift-Cube-OpenArm-Play-v0 --num_envs=1
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0 --num_envs=1
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --num_envs=1
```

---

## 3. Training the LL Policy

The LL task trains a goal-conditioned EE tracker.  The policy observes:
- Robot joint positions and velocities
- Current EE pose in robot base frame
- Commanded target EE pose (sampled uniformly above the table)
- Binary gripper command (sampled once per episode)
- Current gripper opening (normalised)
- Last action

```bash
python scripts/rsl_rl/train.py \
    --task=Nepher-OpenArm-CubeStack-LL-v0 \
    --headless \
    --num_envs=4096
```

Play back a trained LL policy:

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

---

## 4. Exporting the LL Policy

After training, export the checkpoint to `best_policy/`:

```bash
python scripts/rsl_rl/export_policy.py \
    --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

This copies the latest checkpoint to:
```
best_policy/best_policy.pt
best_policy/exported/policy.pt      ← TorchScript
best_policy/exported/policy.onnx    ← ONNX (opset 17)
```

---

## 5. Running the HL Classical Stack Planner

The HL play environment loads the frozen LL policy and drives it with the
classical planner.  The planner progresses through stages:

```
PRE_GRASP → DESCEND → GRASP → LIFT →
MOVE_ABOVE_STACK → LOWER_TO_STACK → RELEASE → RETRACT → NEXT_CUBE → DONE
```

```bash
python scripts/rsl_rl/play.py \
    --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 \
    --checkpoint=best_policy/best_policy.pt \
    --video --video_length=600
```

The play script loads the LL weights, creates the HL env (with classical
planner command term), and applies the LL actor to produce joint actions.

---

## 6. Deterministic Evaluation (30-Scenario Tournament)

Each of the 30 evaluation scenarios fixes:
- Five cube initial positions on the table
- Stack target XY position
- Optional per-cube yaw offsets

Environment `i` uses scenario `i % 30` (deterministic for any number of envs).

```bash
python scripts/eval/evaluate_stack.py \
    --task=Nepher-OpenArm-CubeStack-Eval-v0 \
    --num_envs=30 \
    --episodes=30
```

Results are written to:
```
logs/eval/openarm_cube_stack/<timestamp>/results.json
```

Example output:
```json
{
  "task": "Nepher-OpenArm-CubeStack-Eval-v0",
  "num_envs": 30,
  "episodes": 30,
  "full_stack_success_rate": 0.0,
  "average_cubes_stacked": 0.0,
  "per_cube_success_rate": [0, 0, 0, 0, 0],
  "cube_drop_rate": 0.0,
  "stack_collapse_rate": 0.0,
  "mean_final_stack_error": 0.0,
  "mean_episode_length_s": 0.0,
  "mean_grasp_retries": 0.0,
  "timeout_rate": 0.0
}
```

### Generate a Human-Readable Report

```bash
python scripts/eval/make_eval_report.py \
    logs/eval/openarm_cube_stack/
```

---

## 7. Reproducing 30-Scenario Tournament Scoring

1. Train the LL policy to convergence:
   ```bash
   python scripts/rsl_rl/train.py \
       --task=Nepher-OpenArm-CubeStack-LL-v0 \
       --headless --num_envs=4096 --max_iterations=3000
   ```

2. Export the best checkpoint:
   ```bash
   python scripts/rsl_rl/export_policy.py \
       --task=Nepher-OpenArm-CubeStack-LL-Play-v0
   ```

3. Run the tournament:
   ```bash
   python scripts/eval/evaluate_stack.py \
       --task=Nepher-OpenArm-CubeStack-Eval-v0 \
       --num_envs=30 \
       --episodes=30
   ```

4. View the report:
   ```bash
   python scripts/eval/make_eval_report.py logs/eval/openarm_cube_stack/
   ```

---

## Additional Utility Commands

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
| `average_cubes_stacked` | Mean number of cubes successfully placed per episode |
| `per_cube_success_rate` | Per-cube stacking success rate [cube 1 .. cube 5] |
| `cube_drop_rate` | Fraction of episodes with at least one cube dropping off table |
| `stack_collapse_rate` | Fraction of episodes with stack collapse after placement |
| `mean_final_stack_error` | Mean positional error to target at episode end (m) |
| `mean_episode_length_s` | Mean episode duration (s) |
| `mean_grasp_retries` | Mean number of grasp retries per episode |
| `timeout_rate` | Fraction of episodes that hit the time limit |

A cube is considered **successfully stacked** when:
- XY error to target < 0.04 m
- Height error to expected stack level < 0.02 m
- Linear velocity < 0.05 m/s
- All cubes below it remain stable

---

## License

Copyright (c) 2025-2026, Nepher AI.  
Licensed under the BSD-3-Clause License.  See `LICENSE` for details.
