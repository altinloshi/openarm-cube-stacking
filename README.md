# OpenArm Cube Stacking Task

**Developed by Nepher AI — contact@nepher.ai**

## Overview

This project implements a highly sequential multi-object cube stacking task for the unimanual OpenArm manipulator agent in Isaac Lab. The task trains a continuous control policy to coordinate its continuous arm joints and binary gripper to build a 5-block vertical tower under strict physical stability constraints. 

The policy must execute a multi-stage pick-and-place sequence across progressive operational phases:
1. **Foundation Phase:** Reach, grasp, and position Cube 1 at the designated tabletop base coordinates (Target 1).
2. **Sequential Assembly (Cubes 2, 3, and 4):** Dynamically transition back across the workspace to systematically retrieve, lift, transport, and center each successive cube onto the apex of the growing tower.
3. **Apex Phase:** Retrieve Cube 5, precisely align it, and deposit it at the top of the 4-block structure to finalize the vertical column.

## Requirements

- **Isaac Lab**: 2.3.0
- **Isaac Sim**: 5.1

## Installation

1. Install Isaac Lab following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Install the extension in editable mode:

    ```bash
     python -m pip install -e source/openarmstacking
    ```

3. Verify installation:

    ```bash
    python scripts/list_envs.py
    ```

## Usage

### Training

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0
```

### Playing/Testing

```bash
python scripts/rsl_rl/play.py --task=Nepher-OpenArm-CubeStack-Play-v0 --checkpoint=/path/to/checkpoint.pt
```

### Testing with Random Actions

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0
```

## envhub (nepher) Integration

This project integrates with the [envhub](../../envhub/) (nepher) framework, providing standardized navigation environments with predefined terrains, obstacles, and waypoint configurations.

### Usage

**Training:**
```bash
python scripts/rsl_rl/train.py --task=Nepher-Leatherback-WaypointNav-Envhub-v0
```

**Playing/Testing:**
```bash
python scripts/rsl_rl/play.py --task=Nepher-Leatherback-WaypointNav-Envhub-Play-v0 --checkpoint=/path/to/checkpoint.pt
```

**Customizing scenes:**
```python
from leatherbacknav.tasks.manager_based.waypoint_nav.waypoint_nav_env_cfg_envhub import WaypointNavEnvCfg_Envhub

cfg = WaypointNavEnvCfg_Envhub(scene_id=1)  # Use scene 1
env = gym.make("Nepher-Leatherback-WaypointNav-Envhub-v0", cfg=cfg)
```

## Environment Details

The Leatherback is a 4-wheeled rover-style robot with:
- **Throttle control**: Velocity control for all 4 wheels
- **Steering control**: Position control for front wheel knuckles (Ackermann-style steering)

The action space is 2D: `[throttle, steering]`

See the configuration files in `source/leatherbacknav/leatherbacknav/tasks/manager_based/waypoint_nav/` for full details including observations, actions, rewards, and termination conditions.

## License

This project is licensed under the **BSD-3-Clause License**. See [LICENSE](LICENSE) for details.

Copyright (c) 2025-2026, Nepher AI.

