import math
from typing import Optional, Tuple

import pygame

from engine.entity.transformable import Transformable
from engine.lerp import lerp


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

    def get_coord(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get the coordinate from the camera's perspective.

        :param x: The x coordinate
        :param y: The y coordinate
        :return: The transformed coordinate
        """
        rel_x = x - self.x
        rel_y = y - self.y

        rot_x = rel_x * math.cos(self.rot) - rel_y * math.sin(self.rot)
        rot_y = rel_x * math.sin(self.rot) + rel_y * math.cos(self.rot)

        center = self.screen.get_rect().center

        return rot_x + center[0], rot_y + center[1]

    def update(self, dt: float, follow: Optional[Transformable] = None):
        """
        Update the camera.

        :param dt: The time delta
        :param follow: The new object to follow
        :return: None
        """
        if follow:
            self.follow = follow
            self.x = lerp(self.x, self.follow.x, max(10 * dt, 1))
            self.y = lerp(self.y, self.follow.y, max(10 * dt, 1))
        else:
            self.x = self.follow.x
            self.y = self.follow.y

        self.rot = self.follow.rot
