import math
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
import pygame
import shapely

from engine.entity.camera import Camera
from engine.entity.track import Track
from engine.entity.transformable import Transformable
from engine.utils import clamp, rot_mat, vec


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

        self.color = color
        self.out_of_track_color = out_of_track_color

        self.speed = 0.0
        self.acceleration = 0.0
        self.angular_speed = 0.0
        self.angular_acceleration = 0.0

        self.out_of_track = False
        self.progress = 0
        self.total_progress = 0

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
        self.pos = track.curve.pts[0].pos.astype(np.float32)
        self.rot = np.atan2(*track.get_start_dir()) - math.pi

        self.speed = 0.0
        self.acceleration = 0.0
        self.angular_speed = 0.0
        self.angular_acceleration = 0.0

        self.out_of_track = False
        self.progress = 0
        self.total_progress = len(track.polyline)

    def update(self, dt: float, track: Track):
        """
        Update the car.

        :param dt: The delta time
        :param track: The track
        :return: None
        """
        input_data = self._get_input()

        # Linear movement
        self.acceleration = input_data.forward * self.ACCELERATION * dt

        if self.acceleration != 0:
            self.speed += self.acceleration * dt
            self.speed = clamp(self.speed, -self.MAX_SPEED, self.MAX_SPEED)
        else:
            self.acceleration = (
                self.speed / self.MAX_SPEED * self.DECELERATION * dt
            )
            dspeed = self.acceleration * dt
            if abs(dspeed) > abs(self.speed):
                self.speed = 0
            else:
                self.speed -= dspeed

        # Angular movement
        self.angular_acceleration = (
            input_data.turn * self.ANGULAR_ACCELERATION * dt
        )

        if self.angular_acceleration != 0:
            self.angular_speed += self.angular_acceleration * dt
            self.angular_speed = clamp(
                self.angular_speed,
                -self.MAX_ANGULAR_SPEED,
                self.MAX_ANGULAR_SPEED,
            )
        else:
            self.angular_acceleration = (
                self.angular_speed
                / self.MAX_ANGULAR_SPEED
                * self.ANGULAR_DECELERATION
                * dt
            )
            dangular_speed = self.angular_acceleration * dt
            if abs(dangular_speed) > abs(self.angular_speed):
                self.angular_speed = 0
            else:
                self.angular_speed -= dangular_speed

        # Out of track handling
        self.out_of_track = not track.shapely_polygon.contains(
            shapely.points(self.pos)
        )

        if self.out_of_track:
            self.speed = clamp(
                self.speed,
                -self.MAX_SPEED * self.OUT_OF_TRACK_PENALTY,
                self.MAX_SPEED * self.OUT_OF_TRACK_PENALTY,
            )

        self.translate_forward(self.speed * dt)
        self.rotate(self.angular_speed * dt)

        # Progress
        if self.progress + 1 < len(track.polyline):
            check_point = track.polyline[self.progress + 1]
            if (
                np.linalg.norm(check_point - self.pos)
                < track.width + self.HEIGHT
            ):
                self.progress += 1

    def get_corners(self) -> List[npt.NDArray[np.float32]]:
        """
        Get the transformed corners of the car.

        :return: The transformed corners of the car
        """

        def rotate_angle(
            x: float, y: float, rad: float
        ) -> npt.NDArray[np.float32]:
            pos = vec(x, y)
            return self.pos + np.dot(rot_mat(rad), pos)

        return [
            rotate_angle(-self.WIDTH / 2, -self.HEIGHT / 2, self.rot),
            rotate_angle(self.WIDTH / 2, -self.HEIGHT / 2, self.rot),
            rotate_angle(self.WIDTH / 2, self.HEIGHT / 2, self.rot),
            rotate_angle(-self.WIDTH / 2, self.HEIGHT / 2, self.rot),
        ]

    def draw(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the car.

        :param screen: The screen to draw on
        :param camera: The camera
        :return: None
        """
        polygon = [camera.get_coord(corners) for corners in self.get_corners()]

        pygame.draw.polygon(screen, self.color, polygon)

        if self.out_of_track:
            pygame.draw.polygon(screen, self.out_of_track_color, polygon, 2)
