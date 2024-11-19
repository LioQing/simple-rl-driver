import json
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from engine.activations import ActivationFunc


class CarNN:
    """
    A class representing the neural network of the car.
    """

    prev_fitness: Optional[float]
    prev_weights: Optional[List[npt.NDArray[np.float32]]]
    weights: List[npt.NDArray[np.float32]]
    layer_sizes: List[int]
    activation: ActivationFunc

    def __init__(
        self,
        activation: ActivationFunc,
        weights: Optional[List[np.ndarray]] = None,
        layer_sizes: Optional[List[int]] = None,
    ):
        if not weights and not layer_sizes:
            raise ValueError("Either weights or layer_sizes must be provided")

        self.prev_fitness = None
        self.prev_weights = None

        self.weights = weights or [
            np.random.normal(size=(prev_size + 1, curr_size))
            for prev_size, curr_size in zip(layer_sizes, layer_sizes[1:])
        ]
        self.layer_sizes = layer_sizes or (
            [
                weights[0].shape[0] - 1,
                *(weight.shape[1] for weight in weights),
            ]
        )

        self.activation = activation

    def serialize(self) -> str:
        """
        Serialize the neural network to a string.

        :return: The string
        """
        return json.dumps([weight.tolist() for weight in self.weights])

    @classmethod
    def deserialize(
        cls,
        activation: ActivationFunc,
        string: str,
        init_mutate_noise: float = 0.0,
    ):
        """
        Load the neural network from a string.

        :param activation: The activation function
        :param string: The string
        :param init_mutate_noise: The initial mutation noise
        :return: None
        """
        nn = cls(
            activation=activation,
            weights=[
                np.array(weight, dtype=np.float32)
                for weight in json.loads(string)
            ],
        )

        if init_mutate_noise:
            nn.mutate(init_mutate_noise)

        return nn

    def activate(
        self, inputs: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Activate the neural network.

        :param inputs: The input vector
        :return: The output vector
        """
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} inputs, got {len(inputs)}"
            )

        layer = np.concatenate((inputs, [1.0]))

        for weight in self.weights[:-1]:
            layer = np.concatenate(
                (self.activation(np.dot(layer, weight)), [1.0])
            )

        return np.dot(layer, self.weights[-1])

    def mutate(
        self,
        noise: float,
        learn_rate: float = 0,
        curr_fitness: Optional[float] = None,
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

        sign = np.sign(dfitness)
        for i in range(len(self.weights)):
            self.weights[i] += (
                learn_rate
                * sign
                * dweights[i]
                * np.random.normal(loc=1, scale=noise, size=dweights[i].shape)
            )
