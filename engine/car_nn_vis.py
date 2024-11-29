from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import pygame

from engine.activations import sigmoid


class CarNNVis:
    """
    Visualize the neural network of the car.
    """

    surface: pygame.Surface
    """The surface of the visualization."""

    layer_width: float
    """The width of each layer."""
    node_height: float
    """The height of each node."""
    node_radius: float
    """The radius of each node."""

    node_centers: List[List[Tuple[float, float]]]
    """The centers of the nodes."""
    transforms: List[
        Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
    ]
    """The normalization transforms of the node values."""

    clear_color: pygame.Color
    """The clear color of the surface."""
    node_color: pygame.Color
    """The color of the nodes."""

    def __init__(
        self,
        surface_size: Tuple[int, int],
        layer_sizes: List[int],
        activation: str,
        clear_color: pygame.Color = pygame.Color(32, 32, 32, 192),
        node_color: pygame.Color = pygame.Color(240, 240, 240),
    ):
        """
        Initialize the visualization.

        :param surface_size: The size of the surface
        :param layer_sizes: The sizes of the layers
        :param activation: The activation function
        :param clear_color: The clear color of the surface
        :param node_color: The color of the nodes
        """
        self.surface = pygame.Surface(surface_size, pygame.SRCALPHA)

        self.layer_sizes = layer_sizes
        self.layer_width = self.surface.get_width() / len(layer_sizes)
        self.node_height = self.surface.get_height() / max(layer_sizes)
        self.node_radius = min(self.layer_width, self.node_height) / 3.0

        def identity(x):
            return x

        self.node_centers = [
            [
                (
                    (i + 0.5) * self.layer_width,
                    (j + 0.5 - size * 0.5) * self.node_height
                    + self.surface.get_height() * 0.5,
                )
                for j in range(size)
            ]
            for i, size in enumerate(layer_sizes)
        ]
        self.transforms = [
            identity,
            *(
                [
                    (
                        (lambda x: sigmoid(x) * 2 - 1)
                        if "relu" in activation
                        else identity
                    )
                ]
                * (len(layer_sizes) - 2)
            ),
            identity,
        ]

        self.clear_color = clear_color
        self.node_color = node_color

    def set_weights(self, weights: List[np.ndarray]):
        """
        Compute the surface for the weights, which draws the nodes and lines.

        :param weights: The weights
        :return: None
        """
        if (
            len(weights) + 1 != len(self.layer_sizes)
            or weights[0].shape[0] - 1 != self.layer_sizes[0]
            or any(
                a.shape[1] != b for a, b in zip(weights, self.layer_sizes[1:])
            )
        ):
            raise ValueError(
                "Weights do not match layer sizes:"
                f" {[weights[0].shape[0] - 1] + [w.shape[1] for w in weights]}"
                f" != {self.layer_sizes}"
            )

        self.surface.fill(self.clear_color)

        for i, (from_nodes, to_nodes) in enumerate(
            zip(self.node_centers, self.node_centers[1:])
        ):
            normalized_weights = sigmoid(weights[i]) * 2 - 1
            weight_thickness = (1 + np.abs(normalized_weights) * 5).astype(int)
            weight_colors = self._get_color(weights[i])

            for j, from_node in enumerate(from_nodes):
                for k, to_node in enumerate(to_nodes):
                    pygame.draw.line(
                        self.surface,
                        pygame.Color(int(weight_colors[j, k])),
                        from_node,
                        to_node,
                        weight_thickness[j, k],
                    )

                    pygame.draw.circle(
                        self.surface,
                        self.node_color,
                        from_node,
                        self.node_radius,
                    )

        # draw the last layer
        for nodes in self.node_centers[-1]:
            pygame.draw.circle(
                self.surface, self.node_color, nodes, self.node_radius
            )

    def set_nodes(
        self,
        inputs: npt.NDArray[np.float32],
        hiddens: List[npt.NDArray[np.float32]],
        outputs: npt.NDArray[np.float32],
    ):
        """
        Set the input and output nodes.

        :param inputs: The input nodes
        :param hiddens: The hidden nodes
        :param outputs: The output nodes
        :return: None
        """
        if len(hiddens) + 2 != len(self.node_centers) or any(
            layer.shape[0] != size
            for layer, size in zip(
                [inputs, *hiddens, outputs], self.layer_sizes
            )
        ):
            nodes = [
                inputs.shape[0],
                *(h.shape[0] for h in hiddens),
                outputs.shape[0],
            ]
            raise ValueError(
                "Nodes do not match layer sizes:"
                f" {nodes} != {self.layer_sizes}"
            )

        for nodes, layer_nodes, transform in zip(
            self.node_centers,
            [inputs, *hiddens, outputs],
            self.transforms,
        ):
            for node, color, value in zip(
                nodes,
                self._get_color(layer_nodes),
                np.clip(transform(layer_nodes), -1.0, 1.0),
            ):
                pygame.draw.circle(
                    self.surface,
                    self.node_color,
                    node,
                    self.node_radius,
                )

                pygame.draw.circle(
                    self.surface,
                    pygame.Color(int(color)),
                    node,
                    self.node_radius * abs(value),
                )

    def draw(self, screen: pygame.Surface, coord: Tuple[int, int]):
        """
        Draw the surface on the screen.

        :param screen: The screen
        :param coord: The coordinate
        :return: None
        """
        screen.blit(self.surface, coord or (0, 0))

    @staticmethod
    def _get_color(values: npt.NDArray) -> npt.NDArray[np.int32]:
        """
        Get the color of the values.

        :param values: The values
        :return: The color
        """
        scaled_values = np.sign(values).astype(int) * 255
        green = np.maximum(scaled_values, 0)
        red = np.maximum(-scaled_values, 0)
        return red * 0x01000000 + green * 0x00010000 + 0x000000FF
