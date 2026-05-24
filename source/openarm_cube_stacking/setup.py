from setuptools import find_packages, setup


setup(
    name="openarm-cube-stacking",
    version="0.1.0",
    description="External Isaac Lab task package for OpenArm five-cube stacking.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
