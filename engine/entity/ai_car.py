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
from engine.utils import dir


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
        return self.progress / self.total_progress

    @property
    def sensors(self) -> npt.NDArray[np.float32]:
        """
        Get the sensors of the AI car.

        Equivalent to `self.inputs[2:]`.

        :return: The sensors of the AI car
        """
        return self.inputs[2:]

    @sensors.setter
    def sensors(self, value: npt.NDArray[np.float32]):
        """
        Set the sensors of the AI car.

        Equivalent to setting `self.inputs[2:]`.

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
        # Call the base class constructor, i.e. `Car.__init__`
        super().__init__(
            color=color,
            out_of_track_color=out_of_track_color,
        )

        # Initialize the inputs, which is of length `len(sensor_rots) + 2`
        self.inputs = np.array(
            [0.0] * (len(sensor_rots) + 2), dtype=np.float32
        )

        self.forward = 0.0
        self.turn = 0.0

        self.sensor_rots = sensor_rots
        self.sensor_color = sensor_color

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        self.forward = 0.0
        self.turn = 0.0

        self.sensors.fill(self.SENSOR_DIST)

        super().reset_state(track)

    def update(self, dt: float, track: Track):
        """
        Update the AI car.

        :param dt: The delta time
        :param track: The track
        :return: None
        """
        # Update sensors
        #
        # For each global sensor rotation (the sensor rotation plus the car
        # rotation), find the intersection of those sensor rays with the edge
        # of the track, i.e. `shapely_linear_ring`
        #
        # Then, for each of these intersections, calculate the distance from
        # the car to the intersection, and normalize it by `SENSOR_DIST` so
        # they are within [0, 1]
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
        # Use the sensor rotations and the sensor distances to draw the lines
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
        #
        # Normalize the speed and angular speed by their maximum values so
        # they are within [-1, 1]
        self.inputs[0] = self.speed / self.MAX_SPEED
        self.inputs[1] = self.angular_speed / self.MAX_ANGULAR_SPEED

        # TODO: Activate the neural network to
        # get the ouotput based on the inputs

        # Assign outputs of neural network to inputs of the car
        self.forward = 1
        self.turn = -1

        return Car.Input(self.forward, self.turn)
