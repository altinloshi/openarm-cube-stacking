# Nepher OpenArm Cube Stacking (Isaac Lab External Task)

Manager-based Isaac Lab external task project for:

- `Nepher-OpenArm-CubeStack-v0`
- `Nepher-OpenArm-CubeStack-Play-v0`

The task uses OpenArm to pick and stack five cubes sequentially. This repository is structured like an Isaac Lab external task package with modular task registration, environment configuration, MDP helper files, and RSL-RL scripts.

## Task Summary

- Robot: OpenArm (`OPENARM_UNI_CFG`)
- Control:
  - arm joint position control (`openarm_joint.*`)
  - binary gripper control (`openarm_finger_joint.*`)
- Goal: stack 5 cubes on a target stack base, one cube at a time (current cube = first cube not yet placed).

This first version is a runnable scaffold for the 5-cube setup. For robust learning, use curriculum progression:

1. train on 1 cube
2. then 2 cubes
3. then full 5 cubes

## Setup

1. Install Isaac Lab and Isaac Sim using the official docs.
2. Install this external task package in editable mode:

```bash
python -m pip install -e source/openarm_cube_stacking
```

## Useful Commands

List registered environments:

```bash
python scripts/list_envs.py
```

Run random actions:

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

Run zero actions:

```bash
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

Train with RSL-RL PPO:

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless
```

Play a checkpoint:

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0 --checkpoint=/path/to/model.pt
```

## Repository Layout

Core task code lives in:

`source/openarm_cube_stacking/openarm_cube_stacking/tasks/manager_based/cube_stack/`

including:

- task registration (`__init__.py`)
- train/play env configs
- modular MDP files (`observations.py`, `rewards.py`, `terminations.py`, `events.py`)
- PPO config (`agents/rsl_rl_ppo_cfg.py`)

## Notes

- Uses new Isaac Lab namespace imports (`isaaclab`, `isaaclab_assets`, `isaaclab_tasks`).
- Uses manager-based environment style (`ManagerBasedRLEnv`, `ManagerBasedRLEnvCfg`).
- Includes a simple sequential cube-selection helper and stack target logic suitable as a baseline scaffold.
