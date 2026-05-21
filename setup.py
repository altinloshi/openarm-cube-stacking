from setuptools import setup, find_packages

setup(
    name="openarm_stacking",
    version="0.1.0",
    author="altinloshi",
    description="Autonomous unimanual OpenArm agent for sequential multi-object cube stacking in Isaac Lab.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
    ],
)
