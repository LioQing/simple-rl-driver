import math
from typing import Any

import numpy as np
import numpy.typing as npt


def lerp(a: Any, b: Any, t: Any) -> Any:
    """
    Linear interpolation between a and b by t.

    :param a: The start value.
    :param b: The end value.
    :param t: The interpolation value.
    :return: The interpolated value.
    """
    return a + (b - a) * t


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


def vec2d(
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


def unit_vec_at(rad: float) -> npt.NDArray[np.float32]:
    """
    Get the unit vector at the given angle.
    Note: the engine treat up as 0 radian.
    Also note: the screen's vertical axis is inverted,
    i.e. up is negative y.

    :param rad: The angle in radians.
    :return: The unit vector.
    """
    return vec2d(math.sin(rad), -math.cos(rad))
