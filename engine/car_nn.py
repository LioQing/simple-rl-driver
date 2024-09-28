import json
from typing import List, Union

import numpy as np


class CarNN:
    """
    A class representing the neural network of the car.
    """

    class InputVector:
        """
        A class representing the input vector of the neural network.
        """

        sensors: List[float]

        def __init__(self, rot: float, speed: float, sensors: list[float]):
            self.sensors = sensors

            if len(self.sensors) != 7:
                raise ValueError("Incorrect number of sensors")

        def into_vector(self) -> List[float]:
            """
            Convert the input vector into a list.
            :return: The list
            """
            return [*self.sensors]

    prev_fitness: Union[float, None]
    prev_weights: Union[List[np.ndarray], None]
    weights: List[np.ndarray]

    LAYER_SIZES = (7, 32, 32, 16, 2)

    def __init__(self, prev_fitness: Union[float, None]):
        self.prev_fitness = prev_fitness
        self.prev_weights = None

        self.weights = [
            np.random.normal(size=(prev_size + 1, curr_size))
            for prev_size, curr_size in zip(
                self.LAYER_SIZES, self.LAYER_SIZES[1:]
            )
        ]

    def activate(self, inputs: InputVector) -> List[float]:
        """
        Activate the neural network.
        :param inputs: The input vector
        :return: The output vector
        """

        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

        layer = np.array(inputs.into_vector() + [1.0])

        for weight in self.weights:
            layer = np.dot(layer, weight)
            layer = np.concatenate((relu(layer), [1.0]))

        return list(layer)

    def mutate(
        self,
        noise: float,
        learn_rate: float = 0,
        curr_fitness: Union[float, None] = None,
    ):
        """
        Mutate the neural network.
        :param noise: The noise
        :param learn_rate: The learning rate
        :param curr_fitness: The current fitness
        :return: None
        """
        if (
            self.prev_weights is None
            or curr_fitness is None
            or learn_rate == 0
        ):
            # Random
            for i in range(len(self.weights)):
                self.weights[i] += np.random.normal(
                    loc=0, scale=noise, size=self.weights[i].shape
                )
            return

        # Gradient descent
        dfitness = curr_fitness - self.prev_fitness
        dweights = [
            weight - prev_weight
            for weight, prev_weight in zip(self.weights, self.prev_weights)
        ]

        self.prev_fitness = curr_fitness
        self.prev_weights = self.weights

        for i in range(len(self.weights)):
            sign = np.sign(dfitness if i == 0 else dweights[i - 1])
            self.weights[i] += (
                learn_rate
                * sign
                * dweights[i]
                * np.random.normal(loc=1, scale=noise, size=dweights[i].shape)
            )

    def __str__(self) -> str:
        return "@".join(json.dumps(weight.tolist()) for weight in self.weights)

    def from_string(self, string: str):
        """
        Load the neural network from a string.
        :param string: The string
        :return: None
        """
        weights = string.split("@")
        self.weights = [
            np.array(json.loads(weight)).reshape((prev_size + 1, curr_size))
            for weight, prev_size, curr_size in zip(
                weights, self.LAYER_SIZES, self.LAYER_SIZES[1:]
            )
        ]
