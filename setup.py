"""Top-level setup.py for the openarm-cube-stacking repository.

This file installs the repository-level scripts package only.
The actual Isaac Lab extension lives under source/openarm_cube_stacking/
and must be installed separately:

    python -m pip install -e source/openarm_cube_stacking

See README.md for full installation instructions.
"""

from setuptools import find_packages, setup

setup(
    name="openarm-cube-stacking",
    version="0.1.0",
    author="Nepher",
    description="OpenArm 5-cube sequential stacking – Isaac Lab external task.",
    python_requires=">=3.10",
    # Only discovers packages at the repo root (not inside source/)
    packages=find_packages(exclude=["source", "source.*"]),
    install_requires=[
        "torch",
        "numpy",
        "gymnasium",
    ],
)
