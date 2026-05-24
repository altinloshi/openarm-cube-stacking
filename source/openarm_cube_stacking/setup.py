"""Installation script for the ``openarm_cube_stacking`` Isaac Lab external task."""

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    # core scientific stack; Isaac Lab itself provides the rest
    "torch",
    "numpy",
    "gymnasium",
]

setup(
    name="openarm_cube_stacking",
    version="0.1.0",
    author="Nepher AI",
    author_email="contact@nepher.ai",
    description=(
        "Isaac Lab external manager-based RL task: 5-cube sequential stacking with the OpenArm robot."
    ),
    license="BSD-3-Clause",
    url="https://github.com/nepher-ai/openarm-cube-stacking",
    keywords=["isaaclab", "isaac-lab", "openarm", "robotics", "manipulation", "cube-stacking", "rl"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
