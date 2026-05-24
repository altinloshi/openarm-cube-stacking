"""OpenArm 5-cube stacking environment configuration.

The robot must pick and stack 5 cubes sequentially. The task proceeds cube by cube:
  - cube_0 is placed at the stack base (table surface height)
  - cube_1 is placed on top of cube_0
  - ... up to cube_4 on top of cube_3

The "current cube" is always the lowest-index cube not yet placed at its target.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from . import mdp

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

NUM_CUBES: int = 5
CUBE_SIZE: float = 0.05  # 5 cm side length
CUBE_NAMES: list[str] = [f"cube_{i}" for i in range(NUM_CUBES)]

# Stack base in env-local frame (relative to each env's origin).
# The SeattleLabTable surface sits at ≈ z = 0.03 in world coords when the
# ground plane is at z = -1.05.  We target cube centres at z = 0.055 on the
# table (same as the Isaac-Lift-Cube-OpenArm-v0 reference).
STACK_BASE_LOCAL_X: float = 0.55
STACK_BASE_LOCAL_Y: float = 0.0
STACK_BASE_LOCAL_Z: float = 0.055  # centre of cube_0 when placed

# Initial spawn positions for each cube in env-local frame (non-overlapping,
# all on the table surface, away from the stack target).
_CUBE_INIT_POSITIONS: list[tuple[float, float, float]] = [
    (0.30, -0.20, 0.055),
    (0.30,  0.00, 0.055),
    (0.30,  0.20, 0.055),
    (0.40, -0.20, 0.055),
    (0.40,  0.20, 0.055),
]

# Placement success threshold (distance from target counts as "placed")
PLACE_THRESHOLD: float = 0.03  # 3 cm

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class OpenArmCubeStackSceneCfg(InteractiveSceneCfg):
    """Scene for the 5-cube stacking task."""

    # --- Robot (OpenArm unimanual) ---
    robot: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- End-effector frame sensor ---
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            )
        ],
    )

    # --- Table ---
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # --- Ground plane (positioned so table legs align with it) ---
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # --- Dome light ---
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # --- 5 cubes (procedural rigid bodies, no USD dependency) ---
    cube_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_0",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(_CUBE_INIT_POSITIONS[0]), rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
    )

    cube_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(_CUBE_INIT_POSITIONS[1]), rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
    )

    cube_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(_CUBE_INIT_POSITIONS[2]), rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        ),
    )

    cube_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(_CUBE_INIT_POSITIONS[3]), rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.1)),
        ),
    )

    cube_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(_CUBE_INIT_POSITIONS[4]), rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Action specifications: arm joint positions + binary gripper."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network."""

        # Robot proprioception
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        # End-effector state
        ee_position = ObsTerm(func=mdp.ee_position_in_env_frame)

        # Cube state (all 5, concatenated → [num_envs, 15])
        cube_positions = ObsTerm(func=mdp.cube_positions_in_env_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations)

        # Task progress
        current_cube_index = ObsTerm(func=mdp.current_cube_index_obs)
        current_target_position = ObsTerm(func=mdp.current_target_position_in_env_frame)

        # Relational vectors (most informative for the policy)
        ee_to_current_cube = ObsTerm(func=mdp.ee_to_current_cube)
        current_cube_to_target = ObsTerm(func=mdp.current_cube_to_target)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events (resets)
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Reset events."""

    # Reset the full scene (robot joints + cubes) to their default initial states.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Slightly randomise each cube's position around its default spawn pose.
    # Zero ranges = deterministic reset; widen for curriculum/robustness.
    reset_cube_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )
    reset_cube_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )
    reset_cube_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )
    reset_cube_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_3"),
        },
    )
    reset_cube_4 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_4"),
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Reward terms for the 5-cube stacking MDP."""

    # Shaped reward: exponential decay with EE–cube distance
    reaching_current_cube = RewTerm(
        func=mdp.reaching_current_cube,
        weight=1.0,
        params={"std": 0.1},
    )

    # Bonus for lifting the current cube above table height
    lifting_current_cube = RewTerm(
        func=mdp.lifting_current_cube,
        weight=5.0,
        params={"minimal_height": CUBE_SIZE},
    )

    # Shaped reward: exponential decay with cube–target distance
    moving_current_cube_to_target = RewTerm(
        func=mdp.moving_current_cube_to_target,
        weight=3.0,
        params={"std": 0.2},
    )

    # Fine-grained placing reward (tighter std)
    placing_current_cube = RewTerm(
        func=mdp.placing_current_cube,
        weight=8.0,
        params={"std": 0.05},
    )

    # One-shot bonus when a cube reaches its target position
    cube_placed_bonus = RewTerm(
        func=mdp.cube_placed_bonus,
        weight=15.0,
        params={"threshold": PLACE_THRESHOLD},
    )

    # Episode-level bonus when all 5 cubes are stacked
    stack_success_bonus = RewTerm(
        func=mdp.stack_success_bonus,
        weight=50.0,
        params={"threshold": PLACE_THRESHOLD},
    )

    # Penalty when a cube falls off the table
    cube_drop_penalty = RewTerm(
        func=mdp.cube_drop_penalty,
        weight=-5.0,
        params={"min_height": -0.05},
    )

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
            )
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    all_cubes_stacked = DoneTerm(
        func=mdp.all_cubes_stacked,
        params={"threshold": PLACE_THRESHOLD},
    )

    cube_dropped = DoneTerm(
        func=mdp.cube_dropped,
        params={"min_height": -0.05},
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for Nepher-OpenArm-CubeStack-v0."""

    scene: OpenArmCubeStackSceneCfg = OpenArmCubeStackSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post-init: configure simulation parameters."""
        self.decimation = 2
        self.episode_length_s = 20.0  # longer episodes for 5-cube task

        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
