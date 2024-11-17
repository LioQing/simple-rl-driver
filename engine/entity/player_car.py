import pygame

from engine.entity.car import Car
from engine.utils import lerp


class PlayerCar(Car):
    """
    A class representing the player car.
    """

    forward: float
    turn: float

    def __init__(self):
        super().__init__()
        self.forward = 0
        self.turn = 0

    def _get_input(self) -> Car.Input:
        forward_input = 0
        turn_input = 0

        if pygame.key.get_pressed()[pygame.K_w]:
            forward_input += 1
        if pygame.key.get_pressed()[pygame.K_s]:
            forward_input -= 1

        if forward_input == 0:
            self.forward = 0
        else:
            self.forward = forward_input

        if pygame.key.get_pressed()[pygame.K_a]:
            turn_input -= 1
        if pygame.key.get_pressed()[pygame.K_d]:
            turn_input += 1

        self.turn = lerp(self.turn, turn_input, 0.2)

        self.forward = max(-1.0, min(1.0, self.forward))
        self.turn = max(-1.0, min(1.0, self.turn))

        return Car.Input(self.forward, self.turn)
