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
    """The previous fitness for mutation."""
    prev_weights: Optional[List[npt.NDArray[np.float32]]]
    """The previous weights for mutation."""

    weights: List[npt.NDArray[np.float32]]
    """The weights."""
    layer_sizes: List[int]
    """The sizes of the layers."""
    hiddens: List[npt.NDArray[np.float32]]
    """The hidden layers."""
    activation: ActivationFunc
    """The activation function."""

    def __init__(
        self,
        activation: ActivationFunc,
        weights: Optional[List[np.ndarray]] = None,
        layer_sizes: Optional[List[int]] = None,
    ):
        """
        Initialize the neural network.

        Either `weights` or `layer_sizes` must be provided.

        :param activation: The activation function
        :param weights: The weights of the neural network
        :param layer_sizes: The sizes of the layers
        """
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
        self.hiddens = [
            np.array([0] * size, dtype=np.float32)
            for size in self.layer_sizes[1:-1]
        ]
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

        self.hiddens[0] = self.activation(
            np.dot(np.concatenate((inputs, [1.0])), self.weights[0])
        )

        for i in range(1, len(self.hiddens)):
            self.hiddens[i] = self.activation(
                np.dot(
                    np.concatenate((self.hiddens[i - 1], [1.0])),
                    self.weights[i],
                )
            )

        return np.clip(
            np.dot(
                np.concatenate((self.hiddens[-1], [1.0])),
                self.weights[-1],
            ),
            -1.0,
            1.0,
        )

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
