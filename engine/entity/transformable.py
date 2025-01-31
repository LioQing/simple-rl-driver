import numpy as np
import numpy.typing as npt

from engine.utils import vec


class Transformable:
    """
    A class to represent a Transformable object.
    """

    pos: npt.NDArray[np.float32]
    """Position of the object."""
    rot: float
    """Rotation of the object."""

    def __init__(
        self,
        pos: npt.NDArray[np.float32] = vec(0, 0),
        rot: float = 0,
    ):
        """
        Initialize the Transformable object.

        :param pos: The position of the object
        :param rot: The rotation of the object
        """
        pass

    def translate(self, delta: npt.NDArray[np.float32]):
        """
        Translate the object.

        :param delta: The translation vector
        :return: None
        """
        pass

    def translate_forward(self, dist: float):
        """
        Translate the object forward.

        :param dist: The distance to translate
        :return: None
        """
        pass

    def rotate(self, rad: float):
        """
        Rotate the object.

        :param rad: The rotation in radians
        :return: None
        """
        pass
