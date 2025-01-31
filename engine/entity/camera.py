from typing import Optional

import numpy as np
import numpy.typing as npt
import pygame

from engine.entity.transformable import Transformable
from engine.utils import vec


class Camera(Transformable):
    """
    A class to represent a Camera object.
    """

    follow: Optional[Transformable]
    screen: pygame.Surface

    def __init__(
        self, screen: pygame.Surface, follow: Optional[Transformable] = None
    ):
        super().__init__()

    def get_coord(
        self, pos: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Get the coordinate from the camera's perspective.

        :param pos: The position to transform
        :return: The transformed coordinate
        """
        return vec(0, 0)

    def update(self, dt: float, follow: Optional[Transformable] = None):
        """
        Update the camera.

        :param dt: The time delta
        :param follow: The new object to follow
        :return: None
        """
        pass
