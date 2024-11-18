import math
from typing import Any

import numpy as np
import numpy.typing as npt


def rot_mat(rot: float) -> npt.NDArray[np.float32]:
    """
    Get the rotation matrix for the given rotation.

    :param rot: The rotation in radians.
    :return: The rotation matrix.
    """
    return np.array(
        [
            [math.cos(rot), -math.sin(rot)],
            [math.sin(rot), math.cos(rot)],
        ],
        dtype=np.float32,
    )


def vec(
    x: float, y: float, dtype: Any = np.float32
) -> npt.NDArray[np.float32]:
    """
    Create a 2D vector.

    :param x: The x component.
    :param y: The y component.
    :param dtype: The data type.
    :return: The 2D vector.
    """
    return np.array([x, y], dtype=dtype)


def dir(rad: float) -> npt.NDArray[np.float32]:
    """
    Get the unit vector for the given angle in radians.
    Note: the engine treat up as 0 radian.
    Also note: the screen's vertical axis is inverted,
    i.e. up is negative y.

    :param rad: The angle in radians.
    :return: The unit vector.
    """
    return vec(math.sin(rad), -math.cos(rad))


def clamp(val: Any, min_val: Any, max_val: Any) -> Any:
    """
    Clamp the value between min_val and max_val.

    :param val: The value to clamp.
    :param min_val: The minimum value.
    :param max_val: The maximum value.
    :return: The clamped value.
    """
    return max(min(val, max_val), min_val)
