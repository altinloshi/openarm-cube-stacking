# OpenArm Cube Stacking for Isaac Lab

Developed by Nepher AI — contact@nepher.ai

## Overview

This project implements a cube stacking task for the OpenArm robot in Isaac Lab. The task trains a policy to stack five cubes on top of each other using continuous arm control and binary gripper control.

## Requirements

  * Isaac Lab: 2.3.0
  * Isaac Sim: 5.1

## Installation

  1. Install Isaac Lab following the installation guide.

  2. Install the extension in editable mode:

        python -m pip install -e source/openarm_cube_stacking

  3. Verify installation:

        python scripts/list_envs.py

## Usage

### Training

    python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0

### Playing/Testing

    python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0 --checkpoint=/path/to/checkpoint.pt

### Testing with Random Actions

    python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0

## OpenArm Cube Stacking Integration

This project integrates with the OpenArm cube stacking task, providing a manipulation environment where the robot must pick, move, and stack five cubes on a table.

### Usage

Training:

    python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0

Playing/Testing:

    python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0 --checkpoint=/path/to/checkpoint.pt

Customizing scenes:

    from openarm_cube_stacking.tasks.manager_based.cube_stack.cube_stack_env_cfg import CubeStackEnvCfg
    cfg = CubeStackEnvCfg(num_cubes=5)
    env = gym.make("Nepher-OpenArm-CubeStack-v0", cfg=cfg)

## Environment Details

The OpenArm environment contains a robot arm, a table, and five cubes.

The robot uses:

  * Arm control: Continuous control for reaching, lifting, moving, and placing cubes
  * Gripper control: Binary control for opening and closing the gripper

The action space is composed of arm control and gripper control: `[arm_command, gripper_command]`

The goal of the task is to stack all five cubes vertically on the table. The first cube is placed at the target position, and the remaining cubes are placed one by one on top of it. The final stack must remain stable and upright.

See the configuration files in `source/openarm_cube_stacking/openarm_cube_stacking/tasks/manager_based/cube_stack/` for full details including observations, actions, rewards, and termination conditions.

## License

This project is licensed under the BSD-3-Clause License. See `LICENSE` for details.

Copyright (c) 2025-2026, Nepher AI.
