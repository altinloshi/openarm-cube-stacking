# OpenArm Cube Stacking — Isaac Lab External Task

**Task name:** `Nepher-OpenArm-CubeStack-v0`

An Isaac Lab manager-based reinforcement learning task in which the
[OpenArm](https://github.com/enactic/openarm) unimanual robot must
**pick and stack 5 cubes** sequentially on a table.

---

## Task Description

The robot is placed in front of a table that holds **5 coloured cubes**
(cube_0 through cube_4) in non-overlapping spawn positions.  The objective is
to build a vertical stack at a fixed table location:

| Cube  | Target centre height (env-local z) |
|-------|------------------------------------|
| cube_0 | 0.055 m (table surface level)     |
| cube_1 | 0.105 m (one cube above cube_0)   |
| cube_2 | 0.155 m                            |
| cube_3 | 0.205 m                            |
| cube_4 | 0.255 m                            |

The policy must learn a sequential strategy:

1. Pick **cube_0** and place it at the stack base.
2. Pick **cube_1** and place it on top of cube_0.
3. … repeat through cube_4.

Progress is determined automatically: the *current cube* is the lowest-index
cube not yet within 3 cm of its stack target.

---

## Environment IDs

| ID | Purpose |
|----|---------|
| `Nepher-OpenArm-CubeStack-v0` | Training (4096 envs, observation noise on) |
| `Nepher-OpenArm-CubeStack-Play-v0` | Evaluation (50 envs, noise off) |

---

## Project Structure

```
openarm-cube-stacking/
├── README.md
├── setup.py                          # repo-level placeholder
├── scripts/
│   ├── list_envs.py                  # print registered environments
│   ├── random_agent.py               # random action smoke test
│   ├── zero_agent.py                 # zero action stability test
│   └── rsl_rl/
│       ├── cli_args.py               # shared CLI argument helpers
│       ├── train.py                  # RSL-RL PPO training script
│       └── play.py                   # policy inference / playback
└── source/
    └── openarm_cube_stacking/
        ├── setup.py                  # pip-installable package
        └── openarm_cube_stacking/
            ├── __init__.py
            └── tasks/
                └── manager_based/
                    └── cube_stack/
                        ├── __init__.py        # gym.register()
                        ├── cube_stack_env_cfg.py
                        ├── cube_stack_env_cfg_play.py
                        ├── mdp/
                        │   ├── observations.py
                        │   ├── rewards.py
                        │   ├── terminations.py
                        │   └── events.py
                        └── agents/
                            └── rsl_rl_ppo_cfg.py
```

---

## Requirements

- **Isaac Sim** 5.1 (or compatible)
- **Isaac Lab** 2.3.0
- **Python** 3.10+
- **RSL-RL** (install via `./isaaclab.sh -i rsl_rl`)
- **isaaclab_assets** with OpenArm USD files on NVIDIA Nucleus

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/openarm-cube-stacking.git
cd openarm-cube-stacking

# 2. Install the Isaac Lab extension package (editable install)
python -m pip install -e source/openarm_cube_stacking

# 3. Verify installation
python scripts/list_envs.py
```

> **Note:** Isaac Lab, Isaac Sim, and RSL-RL must already be installed and
> accessible on your Python path.  Follow the
> [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
> before running any scripts here.

---

## Usage

### List registered environments

```bash
python scripts/list_envs.py
```

### Random agent (smoke test)

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

### Zero agent (stability test)

```bash
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

### Train with RSL-RL PPO

```bash
# Headless training (recommended)
python scripts/rsl_rl/train.py \
    --task=Nepher-OpenArm-CubeStack-v0 \
    --headless

# With custom environment count and iteration limit
python scripts/rsl_rl/train.py \
    --task=Nepher-OpenArm-CubeStack-v0 \
    --num_envs=2048 \
    --max_iterations=3000 \
    --headless

# Resume from a previous run
python scripts/rsl_rl/train.py \
    --task=Nepher-OpenArm-CubeStack-v0 \
    --headless \
    --resume \
    --load_run=<run_folder_name>
```

Logs and checkpoints are saved to `logs/rsl_rl/openarm_cube_stack/<timestamp>/`.

### Play (evaluate a trained policy)

```bash
python scripts/rsl_rl/play.py \
    --task=Nepher-OpenArm-CubeStack-Play-v0 \
    --checkpoint=logs/rsl_rl/openarm_cube_stack/<run>/model_5000.pt
```

---

## Architecture Notes

This project follows the
[Isaac Lab extension template](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
pattern:

- **Manager-based RL environment** (`ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`)
- **Modular MDP** split across `mdp/observations.py`, `mdp/rewards.py`,
  `mdp/terminations.py`, and `mdp/events.py`
- **Standard Isaac Lab imports** (`isaaclab.*`, `isaaclab_assets.*`, `isaaclab_rl.*`)
- **Gym registration** via `gymnasium.register()` in the task `__init__.py`
- **RSL-RL PPO** configured through `agents/rsl_rl_ppo_cfg.py`

The OpenArm unimanual configuration (`OPENARM_UNI_CFG`) is imported from
`isaaclab_assets.robots.openarm` following the official Isaac Lab OpenArm
lift-cube task.

---

## Curriculum Recommendations

The full 5-cube task is challenging.  A recommended curriculum:

1. **Phase 1 — Single cube:** Train with 1 cube, short episodes (10 s).
2. **Phase 2 — Two cubes:** Increase to 2 cubes; initialise from Phase 1 checkpoint.
3. **Phase 3 — Five cubes:** Enable all 5 cubes; increase episode length to 20 s.

To implement phases 1–2, reduce `NUM_CUBES` in `cube_stack_env_cfg.py` and
comment out the unused cube scene/event entries, or use a separate env config
class per phase.

---

## Scaffold / TODO

The following features are **scaffolded** with TODO comments and require
further work for production use:

- `mdp/events.py` — `reset_cubes_non_overlapping()` is fully implemented but
  NOT wired into the default `EventCfg` (which uses the simpler
  `reset_root_state_uniform` per cube).  Wire it in for richer randomisation.
- `mdp/events.py` — `reset_stack_target()` is a stub for randomised target
  positions; currently the stack base is fixed.
- `mdp/terminations.py` — `stack_collapsed()` checks cube tilt but does not
  detect dynamic collapse (falling cubes from a partially built stack).
- Curriculum learning across 1 → 5 cubes (described above).
- Sim-to-real transfer considerations (friction tuning, sensor noise modelling).

---

## License

BSD-3-Clause.  See [LICENSE](LICENSE).
