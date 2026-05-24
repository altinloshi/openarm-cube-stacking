# OpenArm Cube Stacking for Isaac Lab

Developed by Nepher AI - contact@nepher.ai

## Overview

This repository is an Isaac Lab external task project for the manager-based RL task:

```text
Nepher-OpenArm-CubeStack-v0
```

The task uses `OPENARM_UNI_CFG` from `isaaclab_assets.robots.openarm` and trains OpenArm to pick and stack five cubes on a table. The implementation follows the Isaac Lab external-task layout with generic scripts, Gymnasium registration, manager-based environment configuration, MDP helper modules, and an RSL-RL PPO runner config.

The play variant is registered as:

```text
Nepher-OpenArm-CubeStack-Play-v0
```

## Requirements

- Isaac Lab with the new `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, and `isaaclab_rl` namespaces
- Isaac Sim compatible with your Isaac Lab version
- Python 3.10+

## Installation

Install Isaac Lab first, then install this external task package in editable mode:

```bash
python -m pip install -e source/openarm_cube_stacking
```

You can also install from the repository root:

```bash
python -m pip install -e .
```

## Verify environment registration

```bash
python scripts/list_envs.py
```

You should see:

```text
Nepher-OpenArm-CubeStack-v0
Nepher-OpenArm-CubeStack-Play-v0
```

## Run agents

Random actions:

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

Zero actions:

```bash
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

## Train with RSL-RL

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless
```

Useful overrides:

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless --num_envs=1024 --max_iterations=5000
```

## Play a checkpoint

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0 --checkpoint=/path/to/model.pt
```

## Project layout

```text
scripts/
  list_envs.py
  random_agent.py
  zero_agent.py
  rsl_rl/
    train.py
    play.py
    cli_args.py

source/openarm_cube_stacking/
  setup.py
  openarm_cube_stacking/
    tasks/manager_based/cube_stack/
      cube_stack_env_cfg.py
      cube_stack_env_cfg_play.py
      mdp/
        observations.py
        rewards.py
        terminations.py
        events.py
      agents/
        rsl_rl_ppo_cfg.py
```

## Task details

The manager-based environment config defines:

- OpenArm joint position control for `openarm_joint.*`
- Binary gripper control for `openarm_finger_joint.*`
- Five rigid cubes with `{ENV_REGEX_NS}` prim paths
- A table, ground plane, lighting, and end-effector frame transformer
- Observations for robot state, last action, end-effector pose, cube poses, current cube index, target stack position, and task vectors
- Reward terms for reaching, lifting, moving to target, placing, full stack success, action penalties, joint velocity penalties, drops, and collapse
- Terminations for timeout, all cubes stacked, cube dropped, and stack collapse
- Reset events for robot joints, cube spawn positions, and stack target position

The sequential stacking logic computes the current cube as the first cube not yet placed at its target. Target centers are:

```text
stack_base + [0, 0, i * cube_size]
```

where cube 0 is centered at tabletop height plus half a cube.

## Scaffold notes

This is a clean runnable scaffold for the full five-cube task. The reward and termination logic is vectorized and structured for training, but five-cube stacking is difficult as a first curriculum. Recommended next steps:

1. Train a one-cube placement curriculum.
2. Extend to two cubes with stronger stability/collapse rewards.
3. Increase to all five cubes once grasping and placement are reliable.

Future improvements can add richer phase-aware rewards, contact-aware grasp detection, domain randomization, curriculum switches, and more robust stack stability metrics.

## License

This project is licensed under the BSD-3-Clause License. See `LICENSE` for details.

Copyright (c) 2025-2026, Nepher AI.
