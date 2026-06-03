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
OpenArm (floor-mounted, official lift-style scene)
```

### Visual Setup

All environments use the **official Isaac-Lift-Cube-OpenArm-Play-v0 scene layout**:

- OpenArm **floor-mounted at env origin** (no table, no workbench).
- Ground plane only.
- Five DexCubes placed in front of the arm at z = 0.055 m (floor-resting height).
- Stack target at x = 0.55 m, z = 0.055 m.
- EE frame: `openarm_link0` → `openarm_ee_tcp`.
- Debug visualisation for EE target frame and planner waypoints (play variants).

Scene constants (`openarm_lift_style_scene_cfg.py`):
```
CUBE_HEIGHT       = 0.055 m   (DexCube edge length, scale 0.8)
CUBE_GROUND_Z     = 0.055 m   (cube centre when resting on floor)
PLANNER_FLOOR_Z   = 0.0275 m  (effective floor z for ClassicalStackPlanner)
```

Cube spawn positions (local env frame):
```
cube_0: (0.35, -0.16, 0.055)
cube_1: (0.35, -0.08, 0.055)
cube_2: (0.40,  0.00, 0.055)
cube_3: (0.35,  0.08, 0.055)
cube_4: (0.35,  0.16, 0.055)
```

Stack target: `(0.55, 0.0, 0.055)` → stacked cube centres at `0.055 + i × 0.055`.

### Environment Overview

| Environment ID | Description |
|---|---|
| `Nepher-OpenArm-CubeStack-v0` | End-to-end baseline (lift-style scene) |
| `Nepher-OpenArm-CubeStack-Play-v0` | Play variant of the baseline |
| `Nepher-OpenArm-CubeStack-EndToEnd-v0` | EndToEnd alias (same as baseline) |
| `Nepher-OpenArm-CubeStack-EndToEnd-Play-v0` | Play variant alias |
| `Nepher-OpenArm-CubeStack-LL-v0` | **Low-level EE tracker** (lift-style scene, no cubes) |
| `Nepher-OpenArm-CubeStack-LL-Play-v0` | Play variant of LL tracker |
| `Nepher-OpenArm-CubeStack-HL-Classical-Play-v0` | **Classical planner + LL policy** (lift-style + cubes) |
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

## Visual Verification

Verify the scene matches the official OpenArm lift layout:

```bash
# Official OpenArm lift scene (reference)
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Isaac-Lift-Cube-OpenArm-Play-v0 --num_envs=1

# LL policy scene (should look identical, no cubes)
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0 --num_envs=1

# HL classical planner scene (should look identical + 5 DexCubes)
../IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --num_envs=1
```

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
    export_policy.py
  eval/
    evaluate_stack.py
    make_eval_report.py

source/openarm_cube_stacking/
  openarm_cube_stacking/
    tasks/manager_based/cube_stack/
      openarm_lift_style_scene_cfg.py  ← canonical scene (robot + ground + DexCubes)
      tabletop_scene_cfg.py            ← backward-compat shim (re-exports from above)
      cube_stack_env_cfg.py            ← end-to-end baseline (lift-style scene)
      cube_stack_env_cfg_play.py
      mdp/                             ← end-to-end MDP modules
      agents/
      end_to_end/                      ← EndToEnd environment aliases
        cube_stack_env_cfg.py
        cube_stack_env_cfg_play.py
        mdp/  agents/
      ll_policy/                       ← goal-conditioned EE tracker
        ll_env_cfg.py
        ll_env_cfg_play.py
        mdp/
          commands.py
          observations.py
          rewards.py
          events.py
          terminations.py
        agents/
          rsl_rl_ppo_cfg.py
      hl_policy/                       ← classical planner + LL execution
        classical_stack_planner.py
        hl_env_cfg.py
        hl_env_cfg_play.py
        hl_env_cfg_eval.py
        mdp/
          commands.py                  ← ClassicalStackPlannerCommandCfg
          observations.py
          events.py
          rewards.py
          terminations.py
        agents/
          rsl_rl_ppo_cfg.py
      eval/                            ← tournament evaluation framework
        scenarios.py                   ← 30 deterministic scenarios
        metrics.py
```

---

## 1. End-to-End Baseline

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

### Lift-Style Scene

All hierarchical environments use the official OpenArm lift-style scene:

```
CUBE_GROUND_Z  = 0.055 m    (cube centre on floor)
CUBE_HEIGHT    = 0.055 m    (DexCube edge length at scale 0.8)
PLANNER_FLOOR_Z = 0.0275 m  (effective floor for the planner formula)
```

Constants live in `openarm_lift_style_scene_cfg.py`.

---

## 3. Training the LL Policy

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

```bash
python scripts/rsl_rl/export_policy.py \
    --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

---

## 5. Running the HL Classical Stack Planner

```bash
python scripts/rsl_rl/play.py \
    --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 \
    --checkpoint=best_policy/best_policy.pt \
    --video --video_length=600
```

---

## 6. Deterministic Evaluation (30-Scenario Tournament)

```bash
python scripts/eval/evaluate_stack.py \
    --task=Nepher-OpenArm-CubeStack-Eval-v0 \
    --num_envs=30 \
    --episodes=30
```

---

## 7. Reproducing 30-Scenario Tournament Scoring

1. Train LL to convergence:
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

---

## License

Copyright (c) 2025-2026, Nepher AI.  
Licensed under the BSD-3-Clause License.  See `LICENSE` for details.
