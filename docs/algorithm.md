# Algorithm — 5-Cube Classical Stack Planner

**Developed by Nepher Robotics — contact@nepher.ai**

---

## Summary

The final, evaluated algorithm for 5-cube stacking is a **classical state-machine planner**
implemented in `ClassicalStackPlanner` (`source/openarm_cube_stacking/openarm_cube_stacking/
tasks/manager_based/cube_stack/hl_policy/classical_stack_planner.py`).

This planner was selected as the reliable final approach.  Learning-based HL policies
(Robomimic, RL HL training) were explored experimentally but are **not** presented as
evaluated results.

---

## Algorithm: ClassicalStackPlanner

### Inputs (per batch step)

| Input | Shape | Description |
|---|---|---|
| `cube_positions` | `(N, 5, 3)` | World-frame XYZ of each cube |
| `ee_pos` | `(N, 3)` | Current EE position (world frame) |
| `ee_quat` | `(N, 4)` | Current EE quaternion (world frame) |
| `dt` | scalar | Physics time step |

### Outputs (per batch step)

| Output | Shape | Description |
|---|---|---|
| `target_pos` | `(N, 3)` | Target EE position for LL policy |
| `target_quat` | `(N, 4)` | Target EE quaternion for LL policy |
| `grip_cmd` | `(N,)` | Binary gripper command (1 = closed) |
| `done` | `(N,)` | Boolean: all cubes placed |

### Stage Transition Logic

```python
# Pseudo-code for a single environment i at stage s
error_pos = || ee_pos[i] - waypoint_pos[i] ||
error_ori = angle_between(ee_quat[i], waypoint_quat[i])

if error_pos < pos_tolerance and error_ori < ori_tolerance:
    dwell_counter[i] += dt
    if dwell_counter[i] >= dwell_time:
        stage[i] = next_stage(s)
        dwell_counter[i] = 0
```

All N environments advance their stages **independently** in parallel.

### Waypoint Computation per Stage

| Stage | Target position | Target quaternion | Gripper |
|---|---|---|---|
| `PRE_GRASP` | cube_i_xy + z_pre_grasp | top-down | open |
| `DESCEND` | cube_i_xyz + z_offset | top-down | open |
| `GRASP` | same (hold) | top-down | closed |
| `LIFT` | cube_i_xy + z_lift | top-down | closed |
| `MOVE_ABOVE_STACK` | stack_xy + z_transport | top-down | closed |
| `LOWER_TO_STACK` | stack_xyz + cube_i × cube_height | top-down | closed |
| `RELEASE` | same (hold) | top-down | open |
| `RETRACT` | stack_xy + z_retract | top-down | open |
| `NEXT_CUBE` | (advance `cube_idx`, loop to PRE_GRASP) | — | open |
| `DONE` | last position (hold) | last orientation | open |

---

## Low-Level Policy Training

The LL policy is trained with PPO in a **separate** environment (`Nepher-OpenArm-CubeStack-LL-v0`)
that has no cubes — only the robot and a randomly sampled EE target.

### Reward Structure

```
r = w_pos_coarse  × exp(-k × ||pos_error||)
  + w_pos_fine    × tanh(s × (1 - ||pos_error||))
  + w_ori         × exp(-k × ori_error)
  + w_grip        × (1 - |grip_cmd - gripper_opening|)
  - w_action_rate × ||action - prev_action||²
  - w_joint_vel   × ||joint_vel||²
```

### Curriculum

Action-rate and joint-velocity penalty weights are ramped from a small initial value
to a higher target over the first N_curriculum iterations to encourage smooth motions
without over-constraining early exploration.

---

## Experimental Approaches (Not Evaluated)

The following HL approaches were explored but are **not the final algorithm**:

### Robomimic / Imitation Learning

- Demonstrations were collected or annotated using Isaac Lab's MimicGen / teleoperation tools.
- Robomimic training configs were prepared.
- This pipeline was **unreliable** and not evaluated at tournament.
- Related config files may be present but are marked as experimental.

### RL-based HL Policy

- An RL-based HL policy was attempted using the `end_to_end` sub-package.
- This approach was abandoned in favour of the classical planner.
- The `end_to_end/` sub-package remains in the repository as a reference baseline.

---

## Why Classical Planning?

1. **Deterministic**: Given correct cube positions, the planner always generates a
   valid grasp sequence.
2. **Interpretable**: Each stage is explicit and debuggable.
3. **No HL training required**: Only the LL policy needs to be trained; the planner
   runs zero-shot.
4. **Parallelisable**: Operates over a full batch of N environments in a single
   vectorised PyTorch pass.
