"""List registered Gymnasium environments for this external task package."""

import gymnasium as gym

import openarm_cube_stacking.tasks  # noqa: F401


def main() -> None:
    env_ids = sorted(str(env_id) for env_id in gym.envs.registry.keys())
    for env_id in env_ids:
        if "OpenArm" in env_id or "CubeStack" in env_id or "Nepher" in env_id:
            print(env_id)


if __name__ == "__main__":
    main()

