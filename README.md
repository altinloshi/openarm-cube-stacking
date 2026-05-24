# OpenArm Cube Stacking — Isaac Lab External Task

Developed by **Nepher AI** — contact@nepher.ai

A clean Isaac Lab *external* manager-based RL task in which the **OpenArm** robot
must pick and stack **five cubes** sequentially.

This repository follows the same project layout used by the official Isaac Lab
manipulation tasks (e.g. ``isaaclab_tasks.manager_based.manipulation.lift``) and
the Nepher external-task convention: a thin top-level project that ships
generic ``scripts/`` and a fully self-contained editable Python package under
``source/openarm_cube_stacking/``.

> The first version is a runnable scaffold. Reward shaping and curriculum are
> intentionally simple and should be tuned for your hardware/compute budget –
> see the *Curriculum* section below.

## Task

| | |
| :-- | :-- |
| Robot | OpenArm (``OPENARM_UNI_CFG`` from ``isaaclab_assets.robots.openarm``) |
| Action | 7 arm joints (``openarm_joint.*``) + binary gripper (``openarm_finger_joint.*``, open=0.044 / close=0.0) |
| Scene | Seattle-lab table, ground plane, dome light, **5 cubes** (``cube_0`` … ``cube_4``) spawned in a non-overlapping row beside the stack base |
| Goal | Sequentially stack ``cube_i`` on top of ``cube_{i-1}`` at a fixed ``stack_base_pos`` |
| Episode length | 12 s |

The "current cube" is computed inside the MDP as the lowest-index cube that has
not yet been placed at its correct slot (see ``mdp.observations``).

## Registered Gym environments

```
Nepher-OpenArm-CubeStack-v0       # training config (4096 envs)
Nepher-OpenArm-CubeStack-Play-v0  # evaluation config (50 envs, no obs noise)
```

Both environments use ``isaaclab.envs:ManagerBasedRLEnv`` and share the same
RSL-RL PPO config (``OpenArmCubeStackPPORunnerCfg``).

## Repository layout

```
openarm-cube-stacking/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   └── rsl_rl/
│       ├── cli_args.py
│       ├── train.py
│       └── play.py
└── source/
    └── openarm_cube_stacking/
        ├── setup.py
        └── openarm_cube_stacking/
            ├── __init__.py
            └── tasks/
                ├── __init__.py
                └── manager_based/
                    ├── __init__.py
                    └── cube_stack/
                        ├── __init__.py            # gym.register(...)
                        ├── cube_stack_env_cfg.py
                        ├── cube_stack_env_cfg_play.py
                        ├── mdp/
                        │   ├── __init__.py
                        │   ├── observations.py
                        │   ├── rewards.py
                        │   ├── terminations.py
                        │   └── events.py
                        └── agents/
                            ├── __init__.py
                            └── rsl_rl_ppo_cfg.py
```

## Requirements

* Isaac Lab ≥ 2.3.0 (uses the new ``isaaclab`` / ``isaaclab_tasks`` /
  ``isaaclab_assets`` namespaces — no legacy ``omni.isaac.lab`` imports).
* Isaac Sim 5.1.
* ``rsl-rl-lib`` (installed by Isaac Lab).

## Installation

1. Install Isaac Lab following the
   [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. From the Isaac Lab Python environment, install this extension in editable mode:

   ```bash
   python -m pip install -e source/openarm_cube_stacking
   ```

## Verifying registration

```bash
python scripts/list_envs.py
```

Expected output (order may vary):

```
Registered OpenArm cube-stacking environments:
  - Nepher-OpenArm-CubeStack-Play-v0
  - Nepher-OpenArm-CubeStack-v0
```

## Running

### Random agent

```bash
python scripts/random_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

### Zero-action agent

```bash
python scripts/zero_agent.py --task=Nepher-OpenArm-CubeStack-v0 --num_envs=4
```

### Training (RSL-RL PPO)

```bash
python scripts/rsl_rl/train.py --task=Nepher-OpenArm-CubeStack-v0 --headless
```

Useful flags: ``--num_envs``, ``--max_iterations``, ``--seed``, ``--resume``,
``--load_run``, ``--checkpoint``, ``--logger``.

### Playing a trained policy

```bash
python scripts/rsl_rl/play.py \
    --task=Nepher-OpenArm-CubeStack-Play-v0 \
    --checkpoint=/path/to/model.pt
```

## Curriculum (recommended)

Stacking five cubes from a flat MDP reward is hard to learn from scratch. We
recommend training with a curriculum:

1. **1 cube** – set ``num_cubes = 1`` and learn pick-and-place to the stack base.
2. **2 cubes** – warm-start from the 1-cube policy with ``num_cubes = 2``.
3. **5 cubes** – fine-tune with the full task.

You can override ``num_cubes`` from your launcher script by setting it on the
env cfg before ``gym.make``:

```python
from openarm_cube_stacking.tasks.manager_based.cube_stack.cube_stack_env_cfg import OpenArmCubeStackEnvCfg
cfg = OpenArmCubeStackEnvCfg()
cfg.num_cubes = 1                  # 1-cube curriculum stage
env = gym.make("Nepher-OpenArm-CubeStack-v0", cfg=cfg)
```

(Note: removing scene cube fields entirely also requires editing
``OpenArmCubeStackSceneCfg`` for that stage.)

## Implementation notes / TODOs

The first version is a *runnable scaffold*. Areas marked for future improvement:

- **Reward shaping**: tune weights, add per-stage gating, possibly per-cube
  targets that switch hard at placement.
- **Stack-collapse termination**: the term ``mdp.stack_collapsed`` is
  implemented but disabled by default until tilt thresholds are tuned.
- **Randomization**: ``reset_cubes_non_overlapping`` and ``reset_stack_target``
  accept ``randomize=True``/``pos_range_xy`` arguments – wire these up once
  basic learning is verified.
- **Curriculum manager**: an Isaac Lab ``CurriculumCfg`` could decay action
  penalties or progressively unlock more cubes – not yet wired up.

## License

BSD-3-Clause. See ``LICENSE`` for details. © 2025-2026 Nepher AI.
