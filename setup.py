from setuptools import find_packages, setup


setup(
    name="openarm-cube-stacking",
    version="0.1.0",
    description="Manager-based Isaac Lab external task for OpenArm five-cube stacking.",
    package_dir={"": "source/openarm_cube_stacking"},
    packages=find_packages(where="source/openarm_cube_stacking"),
    include_package_data=True,
    python_requires=">=3.10",
)
