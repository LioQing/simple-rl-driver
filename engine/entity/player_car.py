import pygame

from engine.entity.car import Car
from engine.utils import clamp


class PlayerCar(Car):
    """
    A class representing the player car.
    """

    def _get_input(self) -> Car.Input:
        return Car.Input(0.0, 0.0)
