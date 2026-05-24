"""Installation script for the openarm_cube_stacking Isaac Lab extension."""

from setuptools import find_packages, setup

setup(
    name="openarm_cube_stacking",
    version="0.1.0",
    author="Nepher",
    description=(
        "Isaac Lab external task: OpenArm 5-cube sequential stacking "
        "(Nepher-OpenArm-CubeStack-v0)."
    ),
    keywords=["robotics", "reinforcement learning", "isaac lab", "openarm", "manipulation"],
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        # Isaac Lab and its asset/RL packages must be installed separately
        # via the Isaac Lab install procedure (isaaclab.sh -i rsl_rl, etc.)
        "torch",
        "numpy",
        "gymnasium",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD Software License",
        "Operating System :: OS Independent",
    ],
)
