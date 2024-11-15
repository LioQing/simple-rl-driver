import math
from typing import List

import pygame
from shapely import LinearRing, LineString, Point

from engine.car_nn import CarNN
from engine.entity.camera import Camera
from engine.entity.car import Car
from engine.entity.track import Track


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
        self.nn = CarNN()
        self.outputs = [0.0, 0.0]

        self.forward = 0
        self.turn = 0
        self.sensors = [self.SENSOR_DIST] * self.SENSOR_COUNT
        self.out_of_track = False

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        self.forward = 0
        self.turn = 0
        self.sensors = [self.SENSOR_DIST] * self.SENSOR_COUNT
        self.out_of_track = False
        super().reset_state(track)

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

    def get_fitness(self) -> float:
        """
        Get the fitness of the AI car.

        :return: The fitness of the AI car
        """
        return self.progress / self.total_progress

    def _get_input(self) -> Car.Input:
        self.outputs = self.nn.activate(
            CarNN.InputVector(
                [sense / self.SENSOR_DIST for sense in self.sensors],
            )
        )

        self.forward = (self.outputs[0] * 2) - 1
        self.turn = (self.outputs[1] * 2) - 1

        self.forward = max(-1.0, min(1.0, self.forward))
        self.turn = max(-1.0, min(1.0, self.turn))

        return Car.Input(self.forward, self.turn)
