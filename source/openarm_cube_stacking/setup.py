from setuptools import find_packages, setup

setup(
    name="openarm_cube_stacking",
    version="0.1.0",
    description="Nepher OpenArm cube stacking external Isaac Lab tasks.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
