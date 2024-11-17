import numpy as np
import numpy.typing as npt
import pygame
from shapely import LinearRing, LineString, Point

from engine.car_nn import CarNN
from engine.entity.camera import Camera
from engine.entity.car import Car
from engine.entity.track import Track
from engine.utils import unit_vec_at, vec2d


class AICar(Car):
    """
    A class representing an AI car.
    """

    forward: float
    turn: float
    sensor_rots: npt.NDArray[np.float32]
    sensors: npt.NDArray[np.float32]
    nn: CarNN
    out_of_track: bool
    outputs: npt.NDArray[np.float32]

    SENSOR_DIST = 500

    def __init__(self, sensor_rots: npt.NDArray[np.float32]):
        super().__init__()
        self.nn = CarNN()
        self.outputs = vec2d(0, 0)
        self.sensor_rots = sensor_rots

        self.forward = 0
        self.turn = 0
        self.sensors = np.array(
            [self.SENSOR_DIST] * len(self.sensor_rots), dtype=np.float32
        )
        self.out_of_track = False

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        self.forward = 0
        self.turn = 0
        self.sensors.fill(self.SENSOR_DIST)
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
        self.sensors.fill(self.SENSOR_DIST)
        self_pos = Point(self.pos)
        track_edges = LinearRing(track.polygon)
        for i, rot in enumerate(self.sensor_rots + self.rot):
            sensor_end = Point(
                self.pos + (unit_vec_at(rot) * self.SENSOR_DIST)
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
        for i, rot in enumerate(self.sensor_rots + self.rot):
            sensor_end = self.pos + (unit_vec_at(rot) * self.sensors[i])
            pygame.draw.line(
                screen,
                pygame.Color(255, 0, 0),
                camera.get_coord(self.pos),
                camera.get_coord(sensor_end),
            )

    def get_fitness(self) -> float:
        """
        Get the fitness of the AI car.

        :return: The fitness of the AI car
        """
        return self.progress / self.total_progress

    def _get_input(self) -> Car.Input:
        self.outputs = self.nn.activate(
            self.sensors / self.SENSOR_DIST,
        )

        self.forward = (self.outputs[0] * 2) - 1
        self.turn = (self.outputs[1] * 2) - 1

        self.forward = max(-1.0, min(1.0, self.forward))
        self.turn = max(-1.0, min(1.0, self.turn))

        return Car.Input(self.forward, self.turn)
