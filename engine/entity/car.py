import math
from dataclasses import dataclass
from typing import List, Tuple

import pygame
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from engine.entity.camera import Camera
from engine.entity.track import Track
from engine.entity.transformable import Transformable


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
    max_speed: float
    turn_speed: float
    out_of_track: bool
    progress: int
    total_progress: int

    DEFAULT_MAX_SPEED = 400
    DEFAULT_TURN_SPEED = 2
    HEIGHT = 32
    WIDTH = 24

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        rot: float = 0,
        max_speed: float = DEFAULT_MAX_SPEED,
        turn_speed: float = DEFAULT_TURN_SPEED,
    ):
        super().__init__(x, y, rot)
        self.speed = 0.0
        self.acceleration = 0.0
        self.max_speed = max_speed
        self.turn_speed = turn_speed
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
        self.x = track.curve.pts[0].x
        self.y = track.curve.pts[0].y
        self.speed = 0.0
        self.acceleration = 0.0
        self.out_of_track = False

        # TODO: Put this into a track method
        direction = track.curve.pts[0].relative_control_point()
        if direction == (0, 0):
            direction = (
                track.curve.pts[1].x - track.curve.pts[0].x,
                track.curve.pts[1].y - track.curve.pts[0].y,
            )

        self.rot = math.atan2(*direction) - math.pi
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
        deceleration = 0.6
        out_of_track_deceleration = 6
        out_of_track_speed = 0.2

        input_data = self._get_input()

        self.acceleration = input_data.forward * 5000 * dt

        if self.acceleration != 0:
            self.speed += self.acceleration * dt
            self.speed = max(-self.max_speed, min(self.max_speed, self.speed))
        else:
            self.speed -= self.speed * deceleration * dt

        self.out_of_track = not Polygon(track.polygon).contains(
            Point(self.x, self.y)
        )

        if (
            abs(self.speed) > self.max_speed * out_of_track_speed
            and self.out_of_track
        ):
            self.speed -= self.speed * out_of_track_deceleration * dt

        self.translate_forward(-self.speed * dt)
        self.rotate(-input_data.turn * self.turn_speed * dt)

        # Progress
        check_point_idx = self.progress + 1
        if check_point_idx < len(track.polyline):
            check_point = Point(track.polyline[check_point_idx])
            if (
                check_point.distance(Point(self.x, self.y))
                < track.width + self.HEIGHT
            ):
                self.progress += 1

    def get_transform(self) -> List[Tuple[float, float]]:
        """
        Get the transformed points of the car.

        :return: The transformed points of the car
        """

        def rotate_angle(
            x: float, y: float, rad: float
        ) -> Tuple[float, float]:
            c = math.cos(rad)
            s = math.sin(rad)
            return self.x + c * x - s * y, self.y + s * x + c * y

        return [
            rotate_angle(-self.WIDTH / 2, -self.HEIGHT / 2, -self.rot),
            rotate_angle(self.WIDTH / 2, -self.HEIGHT / 2, -self.rot),
            rotate_angle(self.WIDTH / 2, self.HEIGHT / 2, -self.rot),
            rotate_angle(-self.WIDTH / 2, self.HEIGHT / 2, -self.rot),
        ]

    def draw(
        self, screen: pygame.Surface, color: pygame.Color, camera: Camera
    ):
        """
        Draw the car.

        :param screen: The screen to draw on
        :param color: The color of the car
        :param camera: The camera
        :return: None
        """
        polygon = [camera.get_coord(x, y) for x, y in self.get_transform()]

        pygame.draw.polygon(screen, color, polygon)
        if self.out_of_track:
            pygame.draw.polygon(screen, pygame.Color(255, 0, 0), polygon, 2)
