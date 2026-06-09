# Copyright (c) 2025-2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Root-level convenience installer — delegates to source/openarm_cube_stacking."""

from setuptools import find_packages, setup

setup(
    name="openarm_cube_stacking",
    version="0.1.0",
    author="Nepher Robotics",
    author_email="contact@nepher.ai",
    description="Isaac Lab external manager-based RL task for OpenArm five-cube stacking.",
    url="https://github.com/nepher-ai/task_franka_cube_stacking",
    package_dir={"": "source/openarm_cube_stacking"},
    packages=find_packages(where="source/openarm_cube_stacking"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "gymnasium",
        "torch",
    ],
    license="BSD-3-Clause",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
