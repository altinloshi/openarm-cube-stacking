from setuptools import find_packages, setup


setup(
    name="openarm_cube_stacking",
    version="0.1.0",
    author="Nepher AI",
    description="Isaac Lab external manager-based RL task for OpenArm five-cube stacking.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "gymnasium",
        "torch",
    ],
)
