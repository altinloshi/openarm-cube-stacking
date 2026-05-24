"""List registered Gym environments relevant to OpenArm cube stacking."""

import gymnasium as gym

import openarm_cube_stacking.tasks  # noqa: F401


def main() -> None:
    keywords = ("OpenArm", "CubeStack", "Nepher")
    env_ids = sorted(env_id for env_id in gym.registry.keys() if any(key in env_id for key in keywords))
    print("Registered environments:")
    for env_id in env_ids:
        print(f" - {env_id}")


if __name__ == "__main__":
    main()
