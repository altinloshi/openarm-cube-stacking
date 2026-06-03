# OpenArm Cube Stacking for Isaac Lab

Developed by Nepher AI - contact@nepher.ai

This repository provides Isaac Lab manager-based tasks for tabletop cube stacking with `OPENARM_UNI_CFG` from `isaaclab_assets.robots.openarm`. The original end-to-end five-cube task is preserved as a baseline, and a new tournament-ready hierarchical pipeline is added beside it.

## Registered environments

List all registered OpenArm tasks:

```bash
python scripts/list_envs.py
```

Expected task IDs include:

```text
Nepher-OpenArm-CubeStack-v0
Nepher-OpenArm-CubeStack-Play-v0
Nepher-OpenArm-CubeStack-EndToEnd-v0
Nepher-OpenArm-CubeStack-EndToEnd-Play-v0
Nepher-OpenArm-CubeStack-LL-v0
Nepher-OpenArm-CubeStack-LL-Play-v0
Nepher-OpenArm-CubeStack-HL-Classical-Play-v0
Nepher-OpenArm-CubeStack-Eval-v0
```

## Visual setup

All new tasks share `tabletop_scene.py`:

- Multiple parallel Isaac Lab environments.
- One large workbench/table per environment.
- OpenArm root initialized on the tabletop via `OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`.
- Five cubes spawned on the same tabletop.
- A stack target on the tabletop.
- Dome light plus debug visualization for the OpenArm EE frame and planner target/stack markers.

The table and Z placement constants are centralized:

```text
TABLE_CENTER
TABLE_SIZE
TABLE_TOP_Z
OPENARM_BASE_Z_OFFSET
ROBOT_BASE_ON_TABLE_Z
ROBOT_BASE_ON_TABLE_POS
CUBE_SIZE
CUBE_TABLE_Z
```

If an OpenArm asset variant has a root frame that is not exactly at the physical mounting surface, tune `OPENARM_BASE_Z_OFFSET` instead of changing each task config.

## Architecture

```text
HL Policy / Classical Stack Planner
    -> waypoints: x, y, z, quat, grip, cube_index
LL Policy / Goal-conditioned EE Tracker
    -> tracks target EE pose + binary gripper command
Differential IK OpenArm action control + binary gripper
    -> OpenArm mounted on tabletop
```

The LL policy is not trained to stack cubes. It only learns reactive EE pose and gripper tracking. The HL classical planner sequences five cubes into a stack by sending static endpoint commands to the frozen LL policy.

## Installation

Install Isaac Lab first, then install this external task package in editable mode:

```bash
python -m pip install -e .
```

The code uses the current Isaac Lab namespaces: `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, and `isaaclab_rl`.

## End-to-end baseline

The original task is still available under its existing IDs:

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0
```

Explicit aliases are also registered:

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-EndToEnd-v0 --headless
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-EndToEnd-Play-v0
```

## Train the LL policy

The LL policy observes robot state, current EE pose in robot base frame, target EE pose, target gripper command, normalized gripper opening, and last action. It uses a 6D relative EE pose command plus a binary gripper command.

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-LL-v0 --headless --num_envs=4096
```

Play the LL policy with debug visualization:

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

If the OpenArm asset build does not expose `openarm_ee_tcp` as a Differential IK-compatible body, `ll_env_cfg.py` includes `FallbackJointPositionActionsCfg` while preserving the same goal-conditioned observation interface.

## Export the LL policy

LL play automatically syncs the latest or requested checkpoint to:

```text
best_policy/best_policy.pt
```

and exports:

```text
best_policy/exported/policy.pt
best_policy/exported/policy.onnx
```

You can also export explicitly:

```bash
python scripts/rsl_rl/export_policy.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

## Run the HL classical stack planner

The HL play task loads the frozen LL checkpoint through the same RSL-RL play path, then the command manager replaces random LL commands with the classical stack planner.

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --video --video_length=600
```

Planner stages:

```text
PRE_GRASP -> DESCEND -> GRASP -> LIFT -> MOVE_ABOVE_STACK -> LOWER_TO_STACK -> RELEASE -> RETRACT -> NEXT_CUBE -> DONE
```

The planner handles five cubes sequentially, advances only after EE pose/orientation tolerances and dwell times are satisfied, and retries missed grasps up to `max_retries`.

## Deterministic tournament evaluation

`Nepher-OpenArm-CubeStack-Eval-v0` uses 30 deterministic scenarios by default:

- scenario index = `env_id % 30`
- fixed cube initial tabletop positions
- fixed stack target
- fixed robot/table placement constants
- no observation noise
- no random object spawn during eval
- reproducible seed

Run evaluation:

```bash
python scripts/eval/evaluate_stack.py --task=Nepher-OpenArm-CubeStack-Eval-v0 --num_envs=30 --episodes=30
```

Results are written to:

```text
logs/eval/openarm_cube_stack/<timestamp>/results.json
```

Metrics include:

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

Create a Markdown report from a result file:

```bash
python scripts/eval/make_eval_report.py logs/eval/openarm_cube_stack/<timestamp>/results.json
```

## Project layout

```text
source/openarm_cube_stacking/openarm_cube_stacking/tasks/manager_based/cube_stack/
  tabletop_scene.py
  end_to_end/
    cube_stack_env_cfg.py
    cube_stack_env_cfg_play.py
    mdp/
    agents/
  ll_policy/
    ll_env_cfg.py
    ll_env_cfg_play.py
    mdp/
    agents/
  hl_policy/
    classical_stack_planner.py
    hl_env_cfg.py
    hl_env_cfg_play.py
    hl_env_cfg_eval.py
    mdp/
    agents/
  eval/
    scenarios.py
    metrics.py

scripts/
  rsl_rl/train.py
  rsl_rl/play.py
  rsl_rl/export_policy.py
  eval/evaluate_stack.py
  eval/make_eval_report.py
```

## Smoke checks

In an Isaac Lab runtime, use:

```bash
python scripts/list_envs.py
python -m compileall source/openarm_cube_stacking
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-LL-v0 --num_envs=2
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --num_envs=2
```

## License

This project is licensed under the BSD-3-Clause License. See `LICENSE` for details.

Copyright (c) 2025-2026, Nepher AI.
