"""``openarm_cube_stacking`` – Isaac Lab external task for OpenArm 5-cube stacking.

Importing this package automatically imports :mod:`openarm_cube_stacking.tasks`,
which registers the Gym environments with :mod:`gymnasium`.
"""

from . import tasks  # noqa: F401  (import for side-effect: gym.register)

__version__ = "0.1.0"
