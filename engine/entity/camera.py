import math
import pygame
from typing import Optional, Tuple
from engine.entity.transformable import Transformable

class Camera(Transformable):
    """
    A class to represent a Camera object.
    """

    car: Optional[Transformable]
    screen: pygame.Surface

    def __init__(self, screen: pygame.Surface, car: Optional[Transformable] = None):
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

        x -= self.x
        y -= self.y

        x, y = x * math.cos(self.rot) - y * math.sin(self.rot), x * math.sin(self.rot) + y * math.cos(self.rot)

        return x + center[0], y + center[1]

    def update(self, dt: float):
        """
        Update the camera.
        :param dt: The time delta
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
            self.x = self.car.x
            self.y = self.car.y
            self.rot = self.car.rot