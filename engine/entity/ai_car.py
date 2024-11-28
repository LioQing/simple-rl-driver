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
from engine.utils import dir, vec


class AICar(Car):
    """
    A class representing an AI car.
    """

    sensor_rots: npt.NDArray[np.float32]
    sensor_color: pygame.Color
    nn: CarNN
    out_of_track: bool

    inputs: npt.NDArray[np.float32]
    outputs: npt.NDArray[np.float32]
    forward: float
    turn: float

    SENSOR_DIST = 500

    @property
    def fitness(self) -> float:
        """
        Get the fitness of the AI car.

        :return: The fitness of the AI car
        """
        return self.progress / self.total_progress

    @property
    def sensors(self) -> npt.NDArray[np.float32]:
        """
        Get the sensors of the AI car.

        :return: The sensors of the AI car
        """
        return self.inputs[2:]

    @sensors.setter
    def sensors(self, value: npt.NDArray[np.float32]):
        """
        Set the sensors of the AI car.

        :param value: The value to set
        :return: None
        """
        self.inputs[2:] = value

    def __init__(
        self,
        sensor_rots: npt.NDArray[np.float32],
        activation: ActivationFunc,
        weights: Optional[str] = None,
        init_mutate_noise: float = 0.0,
        hidden_layer_sizes: Optional[List[int]] = None,
        color: pygame.Color = pygame.Color(0, 0, 0),
        out_of_track_color: pygame.Color = pygame.Color(255, 0, 0),
        sensor_color: pygame.Color = pygame.Color(255, 0, 0),
    ):
        if not weights and not hidden_layer_sizes:
            raise ValueError(
                "Either weights or hidden_layer_sizes must be provided"
            )

        super().__init__(
            color=color,
            out_of_track_color=out_of_track_color,
        )

        self.inputs = np.array(
            [0.0] * (len(sensor_rots) + 2), dtype=np.float32
        )
        self.outputs = vec(0, 0)
        self.forward = 0.0
        self.turn = 0.0

        self.nn = (
            CarNN(
                activation=activation,
                layer_sizes=[len(self.inputs), *hidden_layer_sizes, 2],
            )
            if not weights
            else CarNN.deserialize(
                activation=activation,
                string=weights,
                init_mutate_noise=init_mutate_noise,
            )
        )
        self.sensor_rots = sensor_rots
        self.sensor_color = sensor_color
        self.out_of_track = False

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        self.forward = 0.0
        self.turn = 0.0
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
                    / self.SENSOR_DIST
                    if not intersection.is_empty
                    else 1.0
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
        if not self.out_of_track:
            self.draw_sensor(screen, camera)

        super().draw(screen, camera)

    def draw_sensor(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the sensor of the AI car.

        :param screen: The screen to draw on
        :param camera: The camera
        :return: None
        """
        for rot, dist in zip(
            self.sensor_rots + self.rot,
            self.sensors * self.SENSOR_DIST,
        ):
            sensor_end = self.pos + (dir(rot) * dist)
            pygame.draw.line(
                screen,
                self.sensor_color,
                camera.get_coord(self.pos),
                camera.get_coord(sensor_end),
            )

    def _get_input(self) -> Car.Input:
        # Prepare inputs to the neural network
        self.inputs[0] = self.speed / self.MAX_SPEED
        self.inputs[1] = self.angular_speed / self.MAX_ANGULAR_SPEED

        self.outputs = self.nn.activate(self.inputs)

        self.forward = self.outputs[0]
        self.turn = self.outputs[1]

        return Car.Input(self.forward, self.turn)
