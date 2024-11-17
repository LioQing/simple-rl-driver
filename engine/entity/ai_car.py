import numpy as np
import numpy.typing as npt
import pygame
from shapely import LinearRing, LineString, Point

from engine.car_nn import CarNN
from engine.entity.camera import Camera
from engine.entity.car import Car
from engine.entity.track import Track
from engine.utils import clamp, dir, vec


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

    @property
    def fitness(self) -> float:
        """
        Get the fitness of the AI car.

        :return: The fitness of the AI car
        """
        return self.progress / self.total_progress

    def __init__(self, sensor_rots: npt.NDArray[np.float32]):
        super().__init__()
        self.nn = CarNN()
        self.outputs = vec(0, 0)
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
            sensor_end = Point(self.pos + (dir(rot) * self.SENSOR_DIST))

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
            sensor_end = self.pos + (dir(rot) * self.sensors[i])
            pygame.draw.line(
                screen,
                pygame.Color(255, 0, 0),
                camera.get_coord(self.pos),
                camera.get_coord(sensor_end),
            )

    def _get_input(self) -> Car.Input:
        inputs = np.concatenate(
            (
                self.sensors / self.SENSOR_DIST,
                [self.speed / self.MAX_SPEED],
                [self.angular_speed / self.MAX_ANGULAR_SPEED],
            )
        )
        self.outputs = self.nn.activate(inputs)

        self.forward = clamp(self.outputs[0], -1.0, 1.0)
        self.turn = clamp(self.outputs[1], -1.0, 1.0)

        return Car.Input(self.forward, self.turn)
