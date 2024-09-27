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
    prev_weights: Union[np.ndarray, None]
    prev_weights2: Union[np.ndarray, None]
    prev_weights3: Union[np.ndarray, None]
    prev_weights4: Union[np.ndarray, None]
    weights: np.ndarray
    weights2: np.ndarray
    weights3: np.ndarray
    weights4: np.ndarray

    INPUTS_SIZE = 7
    H1_SIZE = 32
    H2_SIZE = 32
    H3_SIZE = 16
    OUTPUTS_SIZE = 2

    def __init__(self, prev_fitness: Union[float, None]):
        self.prev_fitness = prev_fitness

        self.prev_weights = None
        self.prev_weights2 = None
        self.prev_weights3 = None
        self.prev_weights4 = None

        self.weights = np.random.normal(
            size=(self.INPUTS_SIZE + 1, self.H1_SIZE)
        )
        self.weights2 = np.random.normal(size=(self.H1_SIZE + 1, self.H2_SIZE))
        self.weights3 = np.random.normal(size=(self.H2_SIZE + 1, self.H3_SIZE))
        self.weights4 = np.random.normal(
            size=(self.H3_SIZE + 1, self.OUTPUTS_SIZE)
        )

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

        inputs = np.array(inputs.into_vector() + [1.0])

        layer1 = np.dot(inputs.T, self.weights)
        layer1 = np.concatenate((relu(layer1), [1.0]))

        layer2 = np.dot(layer1.T, self.weights2)
        layer2 = np.concatenate((relu(layer2), [1.0]))

        layer3 = np.dot(layer2.T, self.weights3)
        layer3 = np.concatenate((relu(layer3), [1.0]))

        layer4 = np.dot(layer3.T, self.weights4)
        layer4 = relu(layer4)

        return list(layer4)

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
            # random
            self.weights += np.random.normal(
                loc=0, scale=noise, size=self.weights.shape
            )
            self.weights2 += np.random.normal(
                loc=0, scale=noise, size=self.weights2.shape
            )
            self.weights3 += np.random.normal(
                loc=0, scale=noise, size=self.weights3.shape
            )
            self.weights4 += np.random.normal(
                loc=0, scale=noise, size=self.weights4.shape
            )
            return

        # gradient descent
        dfitness = curr_fitness - self.prev_fitness
        dweights = self.weights - self.prev_weights
        dweights2 = self.weights2 - self.prev_weights2
        dweights3 = self.weights3 - self.prev_weights3
        dweights4 = self.weights4 - self.prev_weights4

        self.prev_fitness = curr_fitness
        self.prev_weights = self.weights
        self.prev_weights2 = self.weights2
        self.prev_weights3 = self.weights3
        self.prev_weights4 = self.weights4

        self.weights += (
            learn_rate
            * np.sign(dfitness)
            * dweights
            * np.random.normal(loc=1, scale=noise, size=dweights.shape)
        )
        self.weights2 += (
            learn_rate
            * np.sign(dweights)
            * dweights2
            * np.random.normal(loc=1, scale=noise, size=dweights2.shape)
        )
        self.weights3 += (
            learn_rate
            * np.sign(dweights2)
            * dweights3
            * np.random.normal(loc=1, scale=noise, size=dweights3.shape)
        )
        self.weights4 += (
            learn_rate
            * np.sign(dweights3)
            * dweights4
            * np.random.normal(loc=1, scale=noise, size=dweights4.shape)
        )

    def __str__(self) -> str:
        return (
            f"{json.dumps(self.weights.tolist())}@"
            f"{json.dumps(self.weights2.tolist())}@"
            f"{json.dumps(self.weights3.tolist())}@"
            f"{json.dumps(self.weights4.tolist())}"
        )

    def from_string(self, string: str):
        """
        Load the neural network from a string.
        :param string: The string
        :return: None
        """
        weights, weights2, weights3, weights4 = string.split("@")
        self.weights = np.array(json.loads(weights)).reshape(
            (self.INPUTS_SIZE + 1, self.H1_SIZE)
        )
        self.weights2 = np.array(json.loads(weights2)).reshape(
            (self.H1_SIZE + 1, self.H2_SIZE)
        )
        self.weights3 = np.array(json.loads(weights3)).reshape(
            (self.H2_SIZE + 1, self.H3_SIZE)
        )
        self.weights4 = np.array(json.loads(weights4)).reshape(
            (self.H3_SIZE + 1, self.OUTPUTS_SIZE)
        )
