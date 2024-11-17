import math

import numpy as np
import numpy.typing as npt

from engine.utils import dir, vec


class Transformable:
    """
    A class to represent a Transformable object.
    """

    pos: npt.NDArray[np.float32]
    rot: float

    def __init__(
        self,
        pos: npt.NDArray[np.float32] = vec(0, 0),
        rot: float = 0,
    ):
        self.pos = pos
        self.rot = rot

    def translate(self, delta: npt.NDArray[np.float32]):
        """
        Translate the object.

        :param delta: The translation vector
        :return: None
        """
        self.pos += delta

    def translate_forward(self, dist: float):
        """
        Translate the object forward.

        :param dist: The distance to translate
        :return: None
        """
        self.translate(dir(self.rot) * dist)

    def rotate(self, rad: float):
        """
        Rotate the object.

        :param rad: The rotation in radians
        :return: None
        """
        self.rot = (self.rot + rad) % (2 * math.pi)
