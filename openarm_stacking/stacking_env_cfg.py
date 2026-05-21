import math
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg

def tracking_phase_machine(env) -> torch.Tensor:
    """Tracks which phase the robot is in (0 to 6) as an observation."""
    if not hasattr(env, "task_phase"):
        env.task_phase = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return env.task_phase.unsqueeze(-1)

def terminate_on_constraint_violation(env) -> torch.Tensor:
    """Enforces rigid stability constraints: 
    1. Cube A must remain stationary at Target A once placed.
    2. Cube B must not tilt more than 5 degrees when stacked.
    """
    cube_a_pos, cube_a_quat = env.scene["cube_a"].data.root_pos_w, env.scene["cube_a"].data.root_quat_w
    cube_b_pos, cube_b_quat = env.scene["cube_b"].data.root_pos_w, env.scene["cube_b"].data.root_quat_w
    target_a_pos = torch.tensor([0.4, 0.0, 0.42], device=env.device)

    # 1. Cube A Displacement Check (Threshold: 2cm away from Target A after placement)
    cube_a_dist_from_target = torch.norm(cube_a_pos[:, :2] - target_a_pos[:2], dim=-1)
    cube_a_moved = (env.task_phase > 2) & (cube_a_dist_from_target > 0.02)

    # 2. Check 5-Degree Tilt Limit on Cube B using Quaternions
    qw, qx, qy, qz = cube_b_quat[:, 0], cube_b_quat[:, 1], cube_b_quat[:, 2], cube_b_quat[:, 3]
    cube_b_z_axis_w = torch.stack([
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx**2 + qy**2)
    ], dim=-1)
    
    global_z = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    dot_product = torch.sum(cube_b_z_axis_w * global_z, dim=-1)
    cube_b_tilted = dot_product < 0.99619 # cos(5 degrees) = 0.99619

    return cube_a_moved | cube_b_tilted

def sequential_stacking_reward(env) -> torch.Tensor:
    """Calculates progressive phase rewards for the sequential task setup."""
    ee_pos = env.scene["robot"].data.ee_pos_w
    cube_a_pos = env.scene["cube_a"].data.root_pos_w
    cube_b_pos = env.scene["cube_b"].data.root_pos_w
    target_a_pos = torch.tensor([0.4, 0.0, 0.42], device=env.device)

    dist_ee_to_cube_a = torch.norm(cube_a_pos - ee_pos, dim=-1)
    dist_cube_a_to_target = torch.norm(target_a_pos - cube_a_pos, dim=-1)
    dist_ee_to_cube_b = torch.norm(cube_b_pos - ee_pos, dim=-1)
    dist_cube_b_to_cube_a = torch.norm(cube_a_pos + torch.tensor([0.0, 0.0, 0.05], device=env.device) - cube_b_pos, dim=-1)

    reward = torch.zeros(env.num_envs, device=env.device)
    env.task_phase = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Phase 0/1: Reach for Cube A
    reward += 1.0 / (1.0 + dist_ee_to_cube_a)
    
    # Phase 2: Lift & Transport Cube A to Target
    is_cube_a_picked = cube_a_pos[:, 2] > 0.42
    env.task_phase[is_cube_a_picked] = 1.0
    reward[is_cube_a_picked] += 2.0 / (1.0 + dist_cube_a_to_target)

    # Phase 3/4: Release A, Move Hand to Cube B
    is_cube_a_placed = dist_cube_a_to_target < 0.02
    env.task_phase[is_cube_a_placed] = 2.0
    reward[is_cube_a_placed] += 1.0 / (1.0 + dist_ee_to_cube_b)

    # Phase 5/6: Stack Cube B directly on top of Cube A
    is_cube_b_picked = is_cube_a_placed & (cube_b_pos[:, 2] > 0.42)
    env.task_phase[is_cube_b_picked] = 3.0
    reward[is_cube_b_picked] += 5.0 / (1.0 + dist_cube_b_to_cube_a)

    return reward

@configclass
class OpenArmSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    table = AssetBaseCfg(prim_path="/World/table", spawn=sim_utils.CuboidCfg(size=(0.8, 1.2, 0.4), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))))
    robot = ArticulationCfg(prim_path="/World/Robot", spawn=sim_utils.UsdFileCfg(usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Franka/franka.usd"))
    cube_a = RigidObjectCfg(prim_path="/World/CubeA", spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), rigid_props=sim_utils.RigidBodyPropertiesCfg(), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))))
    cube_b = RigidObjectCfg(prim_path="/World/CubeB", spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), rigid_props=sim_utils.RigidBodyPropertiesCfg(), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))))

@configclass
class OpenArmCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    scene = OpenArmSceneCfg(num_envs=4096, env_spacing=2.5)

    def __init__(self):
        super().__init__()
        self.sim.dt = 0.01
        self.episode_length_s = 10.0

        self.observations = {
            "policy": ObsGroup(
                terms={
                    "arm_joints": ObsTerm(func="omni.isaac.lab.managers.observation_manager:get_joint_pos"),
                    "cube_a_pos": ObsTerm(func="omni.isaac.lab.managers.observation_manager:get_body_pos", params={"asset_cfg": SceneEntityCfg("cube_a")}),
                    "cube_b_pos": ObsTerm(func="omni.isaac.lab.managers.observation_manager:get_body_pos", params={"asset_cfg": SceneEntityCfg("cube_b")}),
                    "task_phase": ObsTerm(func=tracking_phase_machine),
                }
            )
        }

        self.rewards = {
            "stacking_progress": RewTerm(func=sequential_stacking_reward, weight=10.0),
            "action_penalty": RewTerm(func="omni.isaac.lab.managers.reward_manager:joint_vel_l2", weight=-0.01),
        }

        self.terminations = {
            "time_out": DoneTerm(func="omni.isaac.lab.managers.termination_manager:timeout"),
            "constraint_violation": DoneTerm(func=terminate_on_constraint_violation),
        }
