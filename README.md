# OpenArm Cube Stacking for Isaac Lab

Developed by Nepher AI — contact@nepher.ai

An Isaac Lab external task project for OpenArm tabletop cube stacking. It ships
two pipelines side by side:

1. **End-to-end baseline** — a single RL policy learns the full five-cube stack
   directly from joint-position + binary-gripper actions (the original task,
   preserved as a baseline).
2. **Tournament-ready hierarchical pipeline** — mirrors the
   [task-franka-mani-base](https://github.com/nepher-ai/task-franka-mani-base)
   architecture, adapted to OpenArm and five-cube stacking:

```text
HL classical stack planner
    ↓ sends waypoints: x, y, z, quat, grip, cube_index
LL goal-conditioned EE tracker (RL policy)
    ↓ tracks the target EE pose + binary gripper command
Differential IK (OpenArm openarm_hand) + binary gripper
    ↓
OpenArm robot mounted on the tabletop
```

## Visual setup

All hierarchical-pipeline environments use one **reusable tabletop scene**
(`tabletop.py`) matching the tournament reference:

- a ground plane below a large table/workbench (one per environment),
- the **OpenArm mounted on top of the table** (its base sits on the tabletop
  surface, not on the world floor),
- five cubes spawned on the same tabletop,
- a deterministic stack-target location on the tabletop,
- a dome light,
- an end-effector frame transformer / planner-target debug markers.

All table / robot / cube heights are defined as named constants in one place
(`tabletop.py`), so the layout is easy to tune:

```python
TABLE_CENTER          = (0.55, 0.0, 0.0)
TABLE_TOP_Z           = TABLE_HEIGHT           # tabletop surface height
OPENARM_BASE_Z_OFFSET = 0.0                    # raise if the base penetrates/floats
ROBOT_BASE_ON_TABLE_Z = TABLE_TOP_Z + OPENARM_BASE_Z_OFFSET
ROBOT_BASE_ON_TABLE_POS = (0.55, 0.0, ROBOT_BASE_ON_TABLE_Z)   # robot base on the table
CUBE_SIZE             = 0.05
CUBE_TABLE_Z          = TABLE_TOP_Z + CUBE_SIZE / 2            # cube centre on the table
```

OpenArm body / joint names used throughout (match `OPENARM_UNI_CFG`):
`openarm_joint[1-7]` (arm), `openarm_finger_joint.*` (gripper, open = 0.044 m,
close = 0.0 m), `openarm_link0` (base), `openarm_hand` (end-effector),
`openarm_ee_tcp` (TCP debug frame).

## Requirements

- Isaac Lab with the `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl` namespaces
- Isaac Sim compatible with your Isaac Lab version
- Python 3.10+

## Installation

Install Isaac Lab first, then install this external task package in editable mode:

```bash
python -m pip install -e source/openarm_cube_stacking
# or, from the repository root:
python -m pip install -e .
```

## Registered environments

```bash
python scripts/list_envs.py
```

You should see:

```text
Nepher-OpenArm-CubeStack-v0                    # legacy alias of the end-to-end baseline
Nepher-OpenArm-CubeStack-Play-v0               # legacy alias (play)
Nepher-OpenArm-CubeStack-EndToEnd-v0           # end-to-end baseline (training)
Nepher-OpenArm-CubeStack-EndToEnd-Play-v0      # end-to-end baseline (play)
Nepher-OpenArm-CubeStack-LL-v0                 # low-level EE-tracking policy (training)
Nepher-OpenArm-CubeStack-LL-Play-v0            # low-level EE-tracking policy (play)
Nepher-OpenArm-CubeStack-HL-Classical-Play-v0  # HL classical planner + frozen LL policy
Nepher-OpenArm-CubeStack-Eval-v0               # deterministic 30-scenario tournament
```

## 1. End-to-end baseline

The original task is preserved unchanged under
`tasks/manager_based/cube_stack/end_to_end/`. One policy learns the whole
five-cube stack from joint-position + gripper actions.

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless
python scripts/rsl_rl/play.py  --task=Nepher-OpenArm-CubeStack-Play-v0
```

## 2. Train the low-level (LL) EE-tracking policy

The LL policy learns only to track a commanded end-effector pose and a binary
gripper command (differential IK + binary gripper). It does **not** learn
stacking.

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-LL-v0 --headless --num_envs=4096
python scripts/rsl_rl/play.py  --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

## 3. Export the LL policy

Play and export both copy the latest (or `--checkpoint`) model to
`best_policy/best_policy.pt` and export TorchScript + ONNX to
`best_policy/exported/policy.pt` and `best_policy/exported/policy.onnx`:

```bash
python scripts/rsl_rl/export_policy.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0
```

## 4. Run the HL classical stack planner

The classical planner generates per-cube pick → stack waypoints; the frozen LL
policy executes them. Planner target frames are visualised.

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --video --video_length=600
```

The planner is a vectorised finite state machine with stages:

```text
PRE_GRASP → DESCEND → GRASP → LIFT → MOVE_ABOVE_STACK
    → LOWER_TO_STACK → RELEASE → RETRACT → NEXT_CUBE → (PRE_GRASP | DONE)
```

It handles five cubes sequentially, advances stages only when the EE arrives
within position/orientation tolerance after a minimum dwell (with extra dwell at
GRASP and RELEASE), and retries missed grasps up to `max_retries`. Cube `i` is
stacked at height `TABLE_TOP_Z + CUBE_SIZE/2 + i * CUBE_SIZE`.

## 5. Deterministic tournament evaluation

```bash
python scripts/eval/evaluate_stack.py --task=Nepher-OpenArm-CubeStack-Eval-v0 --num_envs=30 --episodes=30
python scripts/eval/make_eval_report.py   # render the latest results.json as report.md
```

The eval environment runs 30 fully-deterministic scenarios (scenario index =
`env_id % 30`, fixed cube positions, fixed stack target, no observation noise,
reproducible seed). Results are written to:

```text
logs/eval/openarm_cube_stack/<timestamp>/results.json
```

Example `results.json`:

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

A cube counts as stacked when its centre is within position tolerance of the
slot centre, its height matches the expected stack height, it is not moving above
a small velocity threshold, and the cubes below it are also stacked. Full-stack
success requires all five cubes stacked and stable.

## Reproducing the 30-scenario tournament score

```bash
# 1. Train the LL policy.
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-LL-v0 --headless --num_envs=4096

# 2. Export it to best_policy/ (best_policy.pt + exported/policy.pt/.onnx).
python scripts/rsl_rl/export_policy.py --task=Nepher-OpenArm-CubeStack-LL-Play-v0

# 3. Run the deterministic evaluation (30 scenarios, reproducible seed).
python scripts/eval/evaluate_stack.py --task=Nepher-OpenArm-CubeStack-Eval-v0 --num_envs=30 --episodes=30

# 4. Render the report.
python scripts/eval/make_eval_report.py
```

The scenarios are baked once from a fixed seed (`eval/scenarios.py`), so the
evaluation is identical across runs and machines.

## Project layout

```text
scripts/
  list_envs.py
  random_agent.py
  zero_agent.py
  rsl_rl/
    train.py
    play.py            # play + best_policy sync + JIT/ONNX export + video
    export_policy.py   # headless TorchScript/ONNX export
    policy_paths.py    # best_policy checkpoint management
    cli_args.py
  eval/
    evaluate_stack.py    # deterministic tournament eval -> results.json
    make_eval_report.py  # results.json -> report.md

source/openarm_cube_stacking/openarm_cube_stacking/tasks/manager_based/cube_stack/
  tabletop.py            # reusable tabletop scene + all geometry constants
  end_to_end/            # baseline single-policy task (legacy IDs alias this)
    cube_stack_env_cfg.py
    cube_stack_env_cfg_play.py
    mdp/ agents/
  ll_policy/             # low-level EE-tracking policy
    ll_env_cfg.py  ll_env_cfg_play.py
    mdp/ (commands, observations, rewards, events, terminations)  agents/
  hl_policy/             # high-level classical stack planner + env
    classical_stack_planner.py
    hl_env_cfg.py  hl_env_cfg_play.py  hl_env_cfg_eval.py
    mdp/ (commands, observations, events, rewards, terminations)  agents/
  eval/                  # deterministic tournament
    scenarios.py  metrics.py
```

## Run agents (smoke tests)

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-LL-v0 --num_envs=2
python scripts/zero_agent.py   --task=Nepher-OpenArm-CubeStack-HL-Classical-Play-v0 --num_envs=2
```

## Notes on tuning

- `OPENARM_BASE_Z_OFFSET` (in `tabletop.py`) assumes `openarm_link0` sits at the
  physical bottom of the base. If the robot penetrates or floats above the table,
  adjust this single constant.
- The LL `ee_pose` command ranges (`ll_env_cfg.py`) are deliberately broad and
  symmetric to cover the cube/stack region for any base-frame orientation.
  Tighten them to the measured reachable set once validated in simulation.
- The planner `hand_tcp_offset_z` is the vertical distance from a cube grasp
  point up to `openarm_hand`; tune it to match the `openarm_ee_tcp` offset in the
  USD if grasps land high or low.

## License

BSD-3-Clause. See `LICENSE`.

Copyright (c) 2025-2026, Nepher AI.
