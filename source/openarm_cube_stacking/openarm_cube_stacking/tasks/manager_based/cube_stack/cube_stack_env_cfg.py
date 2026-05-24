"""Manager-based environment config for OpenArm 5-cube stacking."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as isaac_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from . import mdp


def _cube_cfg(prim_path: str, init_pos: tuple[float, float, float], color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(init_pos), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(mdp.CUBE_SIZE, mdp.CUBE_SIZE, mdp.CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class OpenArmCubeStackSceneCfg(InteractiveSceneCfg):
    """Scene containing OpenArm, table, and 5 cubes."""

    robot: ArticulationCfg = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    cube_0: RigidObjectCfg = _cube_cfg("{ENV_REGEX_NS}/Cube_0", (0.44, -0.14, mdp.TABLE_TOP_Z + 0.5 * mdp.CUBE_SIZE), (0.9, 0.2, 0.2))
    cube_1: RigidObjectCfg = _cube_cfg("{ENV_REGEX_NS}/Cube_1", (0.44, -0.06, mdp.TABLE_TOP_Z + 0.5 * mdp.CUBE_SIZE), (0.2, 0.9, 0.2))
    cube_2: RigidObjectCfg = _cube_cfg("{ENV_REGEX_NS}/Cube_2", (0.44, 0.02, mdp.TABLE_TOP_Z + 0.5 * mdp.CUBE_SIZE), (0.2, 0.2, 0.9))
    cube_3: RigidObjectCfg = _cube_cfg("{ENV_REGEX_NS}/Cube_3", (0.52, -0.10, mdp.TABLE_TOP_Z + 0.5 * mdp.CUBE_SIZE), (0.9, 0.9, 0.2))
    cube_4: RigidObjectCfg = _cube_cfg("{ENV_REGEX_NS}/Cube_4", (0.52, -0.02, mdp.TABLE_TOP_Z + 0.5 * mdp.CUBE_SIZE), (0.7, 0.3, 0.8))


@configclass
class ActionsCfg:
    """Action terms for OpenArm joint position control + binary gripper."""

    arm_action: isaac_mdp.JointPositionActionCfg = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: isaac_mdp.BinaryJointPositionActionCfg = isaac_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation groups for learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        last_action = ObsTerm(func=isaac_mdp.last_action)
        ee_pose = ObsTerm(func=mdp.ee_pose)
        cube_positions = ObsTerm(func=mdp.cube_positions)
        cube_orientations = ObsTerm(func=mdp.cube_orientations)
        current_cube_index = ObsTerm(func=mdp.current_cube_index)
        current_target_position = ObsTerm(func=mdp.current_target_position)
        ee_to_current_cube = ObsTerm(func=mdp.ee_to_current_cube)
        current_cube_to_target = ObsTerm(func=mdp.current_cube_to_target)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset/randomization events."""

    reset_stack_target = EventTerm(
        func=mdp.reset_stack_target,
        mode="reset",
        params={"xy_noise": 0.015},
    )
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default,
        mode="reset",
        params={"asset_name": "robot", "joint_pos_noise": 0.04},
    )
    reset_cubes = EventTerm(
        func=mdp.reset_cubes_non_overlapping,
        mode="reset",
        params={"xy_noise": 0.015},
    )


@configclass
class RewardsCfg:
    """Reward terms for sequential cube stacking."""

    reaching_current_cube = RewTerm(func=mdp.reaching_current_cube, params={"std": 0.12}, weight=2.0)
    lifting_current_cube = RewTerm(func=mdp.lifting_current_cube, params={"minimal_height": 0.06}, weight=4.0)
    moving_current_cube_to_target = RewTerm(func=mdp.moving_current_cube_to_target, params={"std": 0.16}, weight=6.0)
    placing_current_cube = RewTerm(func=mdp.placing_current_cube, weight=10.0)
    stack_success_bonus = RewTerm(func=mdp.stack_success_bonus, weight=40.0)

    action_penalty = RewTerm(func=mdp.action_penalty, weight=-1.0e-4)
    joint_velocity_penalty = RewTerm(func=mdp.joint_velocity_penalty, weight=-1.0e-4)
    cube_drop_penalty = RewTerm(func=mdp.cube_drop_penalty, weight=-8.0)
    stack_collapse_penalty = RewTerm(func=mdp.stack_collapse_penalty, weight=-6.0)


@configclass
class TerminationsCfg:
    """Termination conditions for stacking episodes."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    all_cubes_stacked = DoneTerm(func=mdp.all_cubes_stacked)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    stack_collapsed = DoneTerm(func=mdp.stack_collapsed)
    invalid_state = DoneTerm(func=mdp.robot_or_cube_invalid_state)


@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """OpenArm cube stacking RL environment."""

    scene: OpenArmCubeStackSceneCfg = OpenArmCubeStackSceneCfg(num_envs=2048, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # Keep contact dynamics stable for stacked cubes.
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
