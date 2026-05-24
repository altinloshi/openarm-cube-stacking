# Nepher OpenArm Cube Stacking

Manager-based Isaac Lab external task project for the OpenArm robot to pick and stack five cubes.

## Project Overview

This repository provides an external Isaac Lab task package named `openarm_cube_stacking` with the
manager-based RL environment IDs:

- `Nepher-OpenArm-CubeStack-v0`
- `Nepher-OpenArm-CubeStack-Play-v0`

The task uses OpenArm joint-position control for the arm and binary open/close control for the
gripper. The first version is intentionally a clean scaffold: it contains the full package layout,
task registration, observation/reward/termination helpers, reset events, RSL-RL PPO config, and
generic train/play/random/zero/list scripts. For best learning performance, a curriculum is still
recommended: start with 1 cube, then 2 cubes, then scale to the full 5-cube stack.

## Requirements

- Isaac Lab with the `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, and `isaaclab_rl` Python packages
- Isaac Sim compatible with your Isaac Lab version
- Python 3.10+

## Installation

1. Install Isaac Lab by following the official setup guide:
   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
2. Install this external task package in editable mode:

   ```bash
   python -m pip install -e source/openarm_cube_stacking
   ```

3. Verify that task registration works:

   ```bash
   python scripts/list_envs.py
   ```

## Usage

### List registered environments

```bash
python scripts/list_envs.py
```

### Run a random policy

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

### Run zero actions

```bash
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

### Train with RSL-RL PPO

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless
```

### Play a trained checkpoint

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0 --checkpoint=/path/to/model.pt
```

## Environment Notes

- The environment is built with `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`.
- OpenArm is configured from `isaaclab_assets.robots.openarm.OPENARM_UNI_CFG`.
- Five cubes are spawned on a tabletop with non-overlapping initial placements.
- Sequential task logic is handled through vectorized MDP helpers that treat the current cube as the
  first cube not yet placed at its target stack position.
- Rewards cover reaching, lifting, moving toward the stack target, placing, successful completion,
  and basic penalties for excessive action, joint motion, drops, and stack collapse.

## Current Scaffold / Recommended Next Steps

This repository is set up as a strong first version, but the following improvements are still
recommended for better training stability and final performance:

- curriculum variants for 1-cube and 2-cube stacking before 5-cube training
- stronger placement/orientation shaping
- richer reset randomization
- additional stack-stability checks and contact-aware rewards
- tuned PPO hyperparameters after initial rollout testing

## License

This project is licensed under the BSD-3-Clause License. See `LICENSE` for details.
