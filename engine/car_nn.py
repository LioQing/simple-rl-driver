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
        pass

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

        # If initial mutation noise is provided, mutate the neural network
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
        pass

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
        pass
