import math
from dataclasses import dataclass
from typing import List, Tuple

import pygame
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing, LineString, Polygon

from engine.car_nn import CarNN
from engine.entity.camera import Camera
from engine.entity.track import Track
from engine.entity.transformable import Transformable
from engine.lerp import lerp


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

    def set_start_pos(self, track: Track):
        """
        Set the start position of the car.
        :param track: The track
        :return: None
        """
        self.x = track.curve.pts[0].x
        self.y = track.curve.pts[0].y
        self.speed = 0.0
        self.acceleration = 0.0
        self.out_of_track = False

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


class AICar(Car):
    """
    A class representing an AI car.
    """

    forward: float
    turn: float
    sensors: List[float]
    nn: CarNN
    out_of_track: bool
    outputs: List[float]

    SENSOR_DIST = 500
    SENSOR_COUNT = 7

    def __init__(self):
        super().__init__()
        self.forward = 0
        self.turn = 0

        # sensor at 0, 45, 90, 135, 180 degrees
        self.sensors = [self.SENSOR_DIST] * self.SENSOR_COUNT

        self.nn = CarNN(None)
        self.out_of_track = False
        self.outputs = [0.0, 0.0]

    def set_start_pos(self, track: Track):
        """
        Set the start position to the first point of the track.
        :param track: The track
        :return: None
        """
        self.forward = 0
        self.turn = 0

        # Sensor at 0, 45, 90, 135, 180 degrees
        self.sensors = [self.SENSOR_DIST] * self.SENSOR_COUNT

        self.out_of_track = False
        super().set_start_pos(track)

    def update(self, dt: float, track: Track):
        """
        Update the AI car.
        :param dt: The delta time
        :param track: The track
        :return: None
        """
        # Update sensors
        self.sensors = [self.SENSOR_DIST] * self.SENSOR_COUNT
        self_pos = Point(self.x, self.y)
        track_edges = LinearRing(track.polygon)
        for i in range(len(self.sensors)):
            angle = -self.rot + math.radians(
                180 + i * 180 / (self.SENSOR_COUNT - 1)
            )
            sensor_end = Point(
                self.x + math.cos(angle) * self.SENSOR_DIST,
                self.y + math.sin(angle) * self.SENSOR_DIST,
            )

            intersection = track_edges.intersection(
                LineString([self_pos, sensor_end])
            )

            if intersection.is_empty:
                continue

            self.sensors[i] = intersection.distance(self_pos)

        super().update(dt, track)

    def draw(
        self, screen: pygame.Surface, color: pygame.Color, camera: Camera
    ):
        """
        Draw the AI car.
        :param screen: The screen to draw on
        :param color: The color of the car
        :param camera: The camera
        :return: None
        """
        super().draw(screen, color, camera)

        if self.out_of_track:
            return

        # Draw sensors lines
        for i in range(len(self.sensors)):
            if self.sensors[i] == float("inf"):
                continue
            angle = -self.rot + math.radians(
                180 + i * 180 / (self.SENSOR_COUNT - 1)
            )
            line = (
                (self.x, self.y),
                (
                    self.x + math.cos(angle) * self.sensors[i],
                    self.y + math.sin(angle) * self.sensors[i],
                ),
            )
            pygame.draw.line(
                screen,
                pygame.Color(255, 0, 0),
                camera.get_coord(*line[0]),
                camera.get_coord(*line[1]),
            )

    def _get_input(self) -> Car.Input:
        self.outputs = self.nn.activate(
            CarNN.InputVector(
                self.rot / (2 * math.pi),
                self.speed / self.max_speed,
                [s / self.SENSOR_DIST for s in self.sensors],
            )
        )

        forward_input = 0
        turn_input = 0

        if self.outputs[0] > 0.7:
            forward_input += 1
        if self.outputs[0] < 0.3:
            forward_input -= 1

        if forward_input == 0:
            self.forward = 0
        else:
            self.forward = forward_input

        if self.outputs[1] > 0.7:
            turn_input -= 1
        if self.outputs[1] < 0.3:
            turn_input += 1

        self.turn = lerp(self.turn, turn_input, 0.2)

        self.forward = max(-1.0, min(1.0, self.forward))
        self.turn = max(-1.0, min(1.0, self.turn))

        return Car.Input(self.forward, self.turn)

    def get_fitness(self) -> float:
        """
        Get the fitness of the AI car.
        :return: The fitness of the AI car
        """
        return self.progress / self.total_progress


class PlayerCar(Car):
    """
    A class representing the player car.
    """

    enabled: bool
    forward: float
    turn: float

    def __init__(self, enabled: bool = True):
        super().__init__(0, 0, 0, 400)
        self.enabled = enabled
        self.forward = 0
        self.turn = 0

    def _get_input(self) -> Car.Input:
        if not self.enabled:
            return Car.Input(0, 0)

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
