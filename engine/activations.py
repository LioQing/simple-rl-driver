from typing import Callable

import numpy as np
import numpy.typing as npt

ActivationFunc = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]


def softmax(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    A softmax activation function.

    :param x: The input
    :return: The output
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    A sigmoid activation function.

    :param x: The input
    :return: The output
    """
    return 1 / (1 + np.exp(-x))


def relu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    A ReLU activation function.

    :param x: The input
    :return: The output
    """
    return np.maximum(0, x)


def leaky_relu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    A leaky ReLU activation function.

    :param x: The input
    :return: The output
    """
    return np.maximum(0.01 * x, x)


activation_funcs = {
    "softmax": softmax,
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu,
}
