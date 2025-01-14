from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pygame

from engine.activations import ActivationFunc
from engine.car_nn import CarNN
from engine.entity.camera import Camera
from engine.entity.car import Car
from engine.entity.track import Track


class AICar(Car):
    """
    A class representing an AI car.
    """

    inputs: npt.NDArray[np.float32]
    """
    The inputs to the neural network, equal to the speed, angular speed, and
    sensors.
    """
    outputs: npt.NDArray[np.float32]
    """The outputs of the neural network."""
    nn: CarNN
    """The neural network."""

    forward: float
    """The forward value."""
    turn: float
    """The turn value."""

    sensor_rots: npt.NDArray[np.float32]
    """A numpy array of sensor rotations, top is 0."""
    sensor_color: pygame.Color
    """The color of the sensor rays."""

    SENSOR_DIST = 500
    """The maximum sensor distance."""

    @property
    def fitness(self) -> float:
        """
        Get the fitness of the AI car.

        Equal to the normalized progress.

        :return: The fitness of the AI car
        """
        return 0.0

    @property
    def sensors(self) -> npt.NDArray[np.float32]:
        """
        Get the sensors of the AI car.

        Equivalent to `self.inputs[2:]`.

        :return: The sensors of the AI car
        """
        return np.array()

    @sensors.setter
    def sensors(self, value: npt.NDArray[np.float32]):
        """
        Set the sensors of the AI car.

        Equivalent to setting `self.inputs[2:]`.

        :param value: The value to set
        :return: None
        """
        pass

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
        """
        Initialize the AI car.

        Either `weights` or `hidden_layer_sizes` must be provided.

        :param sensor_rots: The sensor rotations
        :param activation: The activation function
        :param weights: The weights of the neural network
        :param init_mutate_noise: The initial mutation noise
        :param hidden_layer_sizes: The hidden layer sizes
        :param color: The color of the car
        :param out_of_track_color: The border color of the car when it is out
            of track
        :param sensor_color: The color of the sensor rays
        """
        super().__init__(color, out_of_track_color)

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        super().reset_state(track)

    def update(self, dt: float, track: Track):
        """
        Update the AI car.

        :param dt: The delta time
        :param track: The track
        :return: None
        """
        super().update(dt, track)

    def draw(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the AI car.

        :param screen: The screen to draw on
        :param camera: The camera
        :return: None
        """
        super().draw(screen, camera)

    def draw_sensor(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the sensor of the AI car.

        :param screen: The screen to draw on
        :param camera: The camera
        :return: None
        """
        pass

    def _get_input(self) -> Car.Input:
        return Car.Input(1.0, 1.0)
