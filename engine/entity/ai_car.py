from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pygame
import shapely

from engine.activations import ActivationFunc
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
    sensor_color: pygame.Color

    SENSOR_DIST = 500

    @property
    def fitness(self) -> float:
        """
        Get the fitness of the AI car.

        :return: The fitness of the AI car
        """
        return self.progress / self.total_progress

    def __init__(
        self,
        track: Track,
        sensor_rots: npt.NDArray[np.float32],
        activation: ActivationFunc,
        weights: Optional[str] = None,
        init_mutate_noise: float = 0.0,
        hidden_layer_sizes: Optional[List[int]] = None,
        color: pygame.Color = pygame.Color(0, 0, 0),
        out_of_track_color: pygame.Color = pygame.Color(255, 0, 0),
        sensor_color: pygame.Color = pygame.Color(255, 0, 0),
    ):
        super().__init__(
            color=color,
            out_of_track_color=out_of_track_color,
        )

        if not weights and not hidden_layer_sizes:
            raise ValueError(
                "Either weights or hidden_layer_sizes must be provided"
            )

        self.nn = (
            CarNN(
                activation=activation,
                layer_sizes=[len(sensor_rots) + 2, *hidden_layer_sizes, 2],
            )
            if not weights
            else CarNN.deserialize(
                activation=activation,
                string=weights,
                init_mutate_noise=init_mutate_noise,
            )
        )
        self.outputs = vec(0, 0)
        self.sensor_rots = sensor_rots

        self.forward = 0
        self.turn = 0
        self.sensors = np.array(
            [self.SENSOR_DIST] * len(self.sensor_rots), dtype=np.float32
        )
        self.out_of_track = False

        self.sensor_color = sensor_color

        self.reset_state(track)

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
        self.sensors = np.array(
            [
                (
                    shapely.points(self.pos).distance(intersection)
                    if not intersection.is_empty
                    else self.SENSOR_DIST
                )
                for intersection in (
                    track.shapely_linear_ring.intersection(
                        shapely.linestrings(
                            [
                                self.pos,
                                self.pos + (dir(rot) * self.SENSOR_DIST),
                            ]
                        )
                    )
                    for rot in self.sensor_rots + self.rot
                )
            ],
            dtype=np.float32,
        )

        super().update(dt, track)

    def draw(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the AI car.

        :param screen: The screen to draw on
        :param camera: The camera
        :return: None
        """
        super().draw(screen, camera)

        if self.out_of_track:
            return

        # Draw sensors lines
        for rot, dist in zip(self.sensor_rots + self.rot, self.sensors):
            sensor_end = self.pos + (dir(rot) * dist)
            pygame.draw.line(
                screen,
                self.sensor_color,
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
