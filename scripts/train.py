import gymnasium as gym
import torch
from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import openarm_stacking 

def main():
    print("Initializing OpenArm Multi-Stage Cube Stacking Trainer...")
    env = gym.make("Isaac-OpenArm-CubeStack-v0")
    
    obs, _ = env.reset()
    print("Environment running successfully!")

    for step in range(1000):
        actions = torch.rand(env.num_envs, env.action_space.shape[0], device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 100 == 0:
            print(f"Step {step} | Mean Reward: {torch.mean(rewards).item():.4f}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
