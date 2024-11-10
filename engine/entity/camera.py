import math
from typing import Optional, Tuple

import pygame

from engine.entity.transformable import Transformable
from engine.lerp import lerp


class Camera(Transformable):
    """
    A class to represent a Camera object.
    """

    car: Optional[Transformable]
    screen: pygame.Surface

    def __init__(
        self, screen: pygame.Surface, car: Optional[Transformable] = None
    ):
        super().__init__()
        self.car = car
        self.screen = screen

    def get_coord(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get the coordinate from the camera's perspective.
        :param x: The x coordinate
        :param y: The y coordinate
        :return: The transformed coordinate
        """
        center = self.screen.get_rect().center

        rel_x = x - self.x
        rel_y = y - self.y

        rot_x = rel_x * math.cos(self.rot) - rel_y * math.sin(self.rot)
        rot_y = rel_x * math.sin(self.rot) + rel_y * math.cos(self.rot)

        return rot_x + center[0], rot_y + center[1]

    def update(self, dt: float, lerp_follow: bool = False):
        """
        Update the camera.
        :param dt: The time delta
        :param lerp_follow: Whether to lerp follow the car
        :return: None
        """
        if self.car is None:
            if pygame.key.get_pressed()[pygame.K_w]:
                self.translate_forward(-500 * dt)
            if pygame.key.get_pressed()[pygame.K_s]:
                self.translate_forward(500 * dt)
            if pygame.key.get_pressed()[pygame.K_a]:
                self.translate(-500 * dt, 0)
            if pygame.key.get_pressed()[pygame.K_d]:
                self.translate(500 * dt, 0)
        else:
            if lerp_follow:
                self.x = lerp(self.x, self.car.x, 0.1)
                self.y = lerp(self.y, self.car.y, 0.1)
            else:
                self.x = self.car.x
                self.y = self.car.y
            self.rot = self.car.rot
