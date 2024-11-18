from typing import Optional

import numpy as np
import numpy.typing as npt
import pygame

from engine.entity.transformable import Transformable
from engine.utils import rot_mat, vec


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
        self.follow = follow
        self.screen = screen

    def get_coord(
        self, pos: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Get the coordinate from the camera's perspective.

        :param pos: The position to transform
        :return: The transformed coordinate
        """
        local_pos = pos - self.pos

        rot_pos = np.dot(rot_mat(-self.rot), local_pos)

        center = vec(*self.screen.get_rect().center)

        return center + rot_pos

    def update(self, dt: float, follow: Optional[Transformable] = None):
        """
        Update the camera.

        :param dt: The time delta
        :param follow: The new object to follow
        :return: None
        """
        if follow:
            self.follow = follow

        self.pos = self.follow.pos
        self.rot = self.follow.rot
