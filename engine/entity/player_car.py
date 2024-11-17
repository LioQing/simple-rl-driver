import pygame

from engine.entity.car import Car
from engine.utils import clamp


class PlayerCar(Car):
    """
    A class representing the player car.
    """

    def _get_input(self) -> Car.Input:
        forward = 0
        if pygame.key.get_pressed()[pygame.K_w]:
            forward += 1
        if pygame.key.get_pressed()[pygame.K_s]:
            forward -= 1

        turn = 0
        if pygame.key.get_pressed()[pygame.K_a]:
            turn -= 1
        if pygame.key.get_pressed()[pygame.K_d]:
            turn += 1

        forward = clamp(forward, -1.0, 1.0)
        turn = clamp(turn, -1.0, 1.0)

        return Car.Input(forward, turn)
