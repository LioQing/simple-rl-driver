import math
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
import pygame

from engine.entity.camera import Camera
from engine.entity.track import Track
from engine.entity.transformable import Transformable
from engine.utils import vec


class Car(Transformable):
    """
    A class representing the car in.
    """

    @dataclass
    class Input:
        """
        The input for the car.
        """

        forward: float
        """
        The forward input, in the range [-1, 1], where 1 is forward and -1 is
        backward.
        """
        turn: float
        """
        The turn input, in the range [-1, 1], where 1 is right and -1 is left.
        """

    color: pygame.Color
    """The color of the car."""
    out_of_track_color: pygame.Color
    """The color of the car when it is out of track."""

    speed: float
    """Linear speed of the car."""
    acceleration: float
    """Linear acceleration of the car."""
    angular_speed: float
    """Angular speed of the car."""
    angular_acceleration: float
    """Angular acceleration of the car."""

    out_of_track: bool
    """Whether the car is out of track."""
    progress: int
    """The progress of the car."""
    total_progress: int
    """The total progress of the car."""

    ACCELERATION = 3000
    """Linear acceleration of the car."""
    ANGULAR_ACCELERATION = 50 * math.pi
    """Angular acceleration of the car."""

    DECELERATION = 6000
    """Linear deceleration of the car when there is no input."""
    ANGULAR_DECELERATION = 100 * math.pi
    """Angular deceleration of the car when there is no input."""

    MAX_SPEED = 300
    """Maximum linear speed of the car."""
    MAX_ANGULAR_SPEED = 0.5 * math.pi
    """Maximum angular speed of the car."""

    OUT_OF_TRACK_PENALTY = 0.6
    """Penalty for being out of track."""

    HEIGHT = 32
    """Height of the car."""
    WIDTH = 24
    """Width of the car."""

    def __init__(
        self,
        pos: npt.NDArray[np.float32] = vec(0, 0),
        rot: float = 0,
        color: pygame.Color = pygame.Color(0, 0, 0),
        out_of_track_color: pygame.Color = pygame.Color(255, 0, 0),
    ):
        """
        Initialize the car.

        :param pos: The position of the car
        :param rot: The rotation of the car
        :param color: The color of the car
        :param out_of_track_color: The border color of the car when it is out
            of track
        """
        super().__init__(pos, rot)

    def _get_input(self) -> Input:
        """
        Get the input for the car.

        :return: The input for the car
        """
        raise NotImplementedError(self._get_input)

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        pass

    def update(self, dt: float, track: Track):
        """
        Update the car.

        :param dt: The delta time
        :param track: The track
        :return: None
        """
        pass

    def get_corners(self) -> List[npt.NDArray[np.float32]]:
        """
        Get the transformed corners of the car.

        :return: The transformed corners of the car
        """
        return []

    def draw(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the car.

        :param screen: The screen to draw on
        :param camera: The camera
        :return: None
        """
        pass
