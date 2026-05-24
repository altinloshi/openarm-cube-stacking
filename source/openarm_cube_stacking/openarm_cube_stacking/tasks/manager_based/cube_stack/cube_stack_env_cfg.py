"""Manager-based environment configuration for the OpenArm 5-cube stacking task.

This config follows the same patterns as the official Isaac Lab manipulation
tasks (in particular ``isaaclab_tasks.manager_based.manipulation.lift``). The
robot, end-effector frame, action set and gripper command values mirror the
official OpenArm single-cube lift configuration so behaviour stays consistent
across tasks.
"""

from __future__ import annotations

from dataclasses import MISSING

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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from . import mdp

##
# Constants
##

NUM_CUBES: int = 5
CUBE_SIZE: float = 0.05  # meters (per side)
TABLE_HEIGHT: float = 0.0  # the SeattleLabTable USD's top surface aligns with z=0 by convention
STACK_BASE_POS: tuple[float, float, float] = (0.5, 0.0, TABLE_HEIGHT)
PLACE_THRESHOLD: float = 0.03  # meters – within which a cube counts as "placed"

# Per-cube RGB tints for visual distinction.
_CUBE_COLORS = [
    (0.9, 0.1, 0.1),
    (0.1, 0.9, 0.1),
    (0.1, 0.1, 0.9),
    (0.9, 0.9, 0.1),
    (0.6, 0.1, 0.9),
]


def _make_cube_cfg(index: int) -> RigidObjectCfg:
    """Build a :class:`RigidObjectCfg` for cube ``index`` using ``{ENV_REGEX_NS}``."""
    color = _CUBE_COLORS[index % len(_CUBE_COLORS)]
    # deterministic non-overlapping initial offset (mirrors mdp.events._grid_offsets)
    pitch = CUBE_SIZE + 0.02
    y = (index - (NUM_CUBES - 1) / 2.0) * pitch
    init_pos = (STACK_BASE_POS[0] - 2.0 * CUBE_SIZE, STACK_BASE_POS[1] + y, STACK_BASE_POS[2] + 0.5 * CUBE_SIZE)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Cube_{index}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=0.5),
        ),
    )


##
# Scene
##


@configclass
class OpenArmCubeStackSceneCfg(InteractiveSceneCfg):
    """Scene with the OpenArm robot, a table, a ground plane and ``NUM_CUBES`` cubes."""

    # Populated in ``__post_init__`` of the env cfg.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # Table (Seattle lab table from Isaac assets).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Five cubes – generated as fields ``cube_0`` .. ``cube_4``.
    cube_0: RigidObjectCfg = _make_cube_cfg(0)
    cube_1: RigidObjectCfg = _make_cube_cfg(1)
    cube_2: RigidObjectCfg = _make_cube_cfg(2)
    cube_3: RigidObjectCfg = _make_cube_cfg(3)
    cube_4: RigidObjectCfg = _make_cube_cfg(4)


##
# MDP terms
##


@configclass
class ActionsCfg:
    """Joint-position arm control + binary gripper, matching the official OpenArm lift task."""

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


@configclass
class ObservationsCfg:
    """Observations exposed to the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group – concatenated by Isaac Lab into a flat vector."""

        # Robot proprioception.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        # End-effector pose.
        ee_pos = ObsTerm(func=mdp.ee_position)

        # Cube state.
        cube_positions = ObsTerm(func=mdp.cube_positions)
        cube_orientations = ObsTerm(func=mdp.cube_orientations)

        # Sequential-stacking signals.
        current_cube_idx = ObsTerm(func=mdp.current_cube_index)
        current_cube_pos = ObsTerm(func=mdp.current_cube_position)
        current_target_pos = ObsTerm(func=mdp.current_target_position)
        ee_to_current_cube = ObsTerm(func=mdp.ee_to_current_cube)
        current_cube_to_target = ObsTerm(func=mdp.current_cube_to_target)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset-time events."""

    # NOTE: The order matters – first reset the robot, then the stack target, then cubes.
    reset_robot = EventTerm(func=mdp.reset_robot_to_default, mode="reset")
    reset_stack_target = EventTerm(
        func=mdp.reset_stack_target,
        mode="reset",
        params={"base_pos": STACK_BASE_POS, "pos_range_xy": (0.0, 0.0)},
    )
    reset_cubes = EventTerm(
        func=mdp.reset_cubes_non_overlapping,
        mode="reset",
        params={"randomize": False, "pos_noise": 0.0},
    )


@configclass
class RewardsCfg:
    """Reward terms encouraging the sequential stacking behaviour."""

    reaching_current_cube = RewTerm(
        func=mdp.reaching_current_cube,
        params={"std": 0.1},
        weight=1.0,
    )
    lifting_current_cube = RewTerm(
        func=mdp.lifting_current_cube,
        params={"minimal_height": TABLE_HEIGHT + 0.05},
        weight=10.0,
    )
    moving_current_cube_to_target = RewTerm(
        func=mdp.moving_current_cube_to_target,
        params={"std": 0.3, "minimal_height": TABLE_HEIGHT + 0.05},
        weight=8.0,
    )
    placing_current_cube = RewTerm(
        func=mdp.placing_current_cube,
        params={"std": 0.05, "minimal_height": TABLE_HEIGHT + 0.05},
        weight=5.0,
    )
    stack_success_bonus = RewTerm(
        func=mdp.stack_success_bonus,
        params={"place_threshold": PLACE_THRESHOLD},
        weight=20.0,
    )
    cube_drop_penalty = RewTerm(
        func=mdp.cube_drop_penalty,
        params={"drop_height": -0.1},
        weight=-5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


@configclass
class TerminationsCfg:
    """Episode-termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.all_cubes_stacked, params={"place_threshold": PLACE_THRESHOLD})
    cube_dropped = DoneTerm(func=mdp.cube_dropped, params={"drop_height": -0.1})
    # TODO: re-enable once tilt thresholds are tuned for the OpenArm scene
    # stack_collapsed = DoneTerm(func=mdp.stack_collapsed, params={"tilt_cosine": 0.9})


##
# Top-level environment configuration
##


@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level configuration for the 5-cube OpenArm stacking task."""

    # Task-level constants exposed for MDP helpers.
    num_cubes: int = NUM_CUBES
    cube_size: float = CUBE_SIZE
    stack_base_pos: tuple[float, float, float] = STACK_BASE_POS
    place_threshold: float = PLACE_THRESHOLD

    # Scene & MDP managers.
    scene: OpenArmCubeStackSceneCfg = OpenArmCubeStackSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # General settings.
        self.decimation = 2
        self.episode_length_s = 12.0  # 5 cubes – longer than single-object lift
        # Simulation settings.
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Robot.
        self.scene.robot = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # End-effector frame transformer (matches the official OpenArm lift config).
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                    name="end_effector",
                ),
            ],
        )
