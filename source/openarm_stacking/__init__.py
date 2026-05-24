import gymnasium as gym
from .stacking_env_cfg import OpenArmCubeStackEnvCfg

# Register your custom multi-stage stacking environment
gym.register(
    id="Isaac-OpenArm-CubeStack-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": OpenArmCubeStackEnvCfg(),
    },
)
