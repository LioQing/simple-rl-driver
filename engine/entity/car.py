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
        turn: float

    speed: float
    acceleration: float
    angular_speed: float
    angular_acceleration: float
    out_of_track: bool
    progress: int
    total_progress: int
    color: pygame.Color
    out_of_track_color: pygame.Color

    ACCELERATION = 3000
    ANGULAR_ACCELERATION = 50 * math.pi

    DECELERATION = 6000
    ANGULAR_DECELERATION = 100 * math.pi

    MAX_SPEED = 300
    MAX_ANGULAR_SPEED = 0.5 * math.pi

    OUT_OF_TRACK_PENALTY = 0.6

    HEIGHT = 32
    WIDTH = 24

    def __init__(
        self,
        pos: npt.NDArray[np.float32] = vec(0, 0),
        rot: float = 0,
        color: pygame.Color = pygame.Color(0, 0, 0),
        out_of_track_color: pygame.Color = pygame.Color(255, 0, 0),
    ):
        super().__init__(pos, rot)
        self.speed = 0.0
        self.acceleration = 0.0
        self.angular_speed = 0.0
        self.angular_acceleration = 0.0
        self.out_of_track = False
        self.progress = 0
        self.total_progress = 0
        self.color = color
        self.out_of_track_color = out_of_track_color

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
        self.speed = 0.0
        self.acceleration = 0.0
        self.angular_speed = 0.0
        self.angular_acceleration = 0.0
        self.out_of_track = False
        self.rot = np.atan2(*track.get_start_dir()) - math.pi
        self.progress = 0
        self.total_progress = len(track.polyline)

    def update(self, dt: float, track: Track):
        """
        Update the car.

        :param dt: The delta time
        :param track: The track
        :return: None
        """
        # Movement
        input_data = self._get_input()

        self.acceleration = input_data.forward * self.ACCELERATION * dt
        self.angular_acceleration = (
            input_data.turn * self.ANGULAR_ACCELERATION * dt
        )

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
