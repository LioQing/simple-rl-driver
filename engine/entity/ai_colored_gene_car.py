from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pygame

from engine.activations import ActivationFunc, sigmoid
from engine.entity.ai_car import AICar
from engine.entity.camera import Camera
from engine.entity.track import Track
from engine.utils import vec


class AIColoredGeneCar(AICar):
    """
    A class representing an AI car with colored weights.
    """

    weights_surf: List[pygame.Surface]
    colored_gene_surf: pygame.Surface

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
        super().__init__(
            sensor_rots=sensor_rots,
            activation=activation,
            weights=weights,
            init_mutate_noise=init_mutate_noise,
            hidden_layer_sizes=hidden_layer_sizes,
            color=color,
            out_of_track_color=out_of_track_color,
            sensor_color=sensor_color,
        )

        self.weights_surf = [
            pygame.Surface((weight.shape[0], weight.shape[1]))
            for weight in self.nn.weights
        ]
        self.colored_gene_surf = pygame.Surface((self.WIDTH, self.HEIGHT))

    def reset_state(self, track: Track):
        """
        Reset the state of the car.

        :param track: The track
        :return: None
        """
        # Update weights surface
        for i, weights in enumerate(self.nn.weights):
            scaled_weights = (sigmoid(weights) * 510 - 255).astype(int)
            green = np.maximum(scaled_weights, 0)
            red = np.maximum(-scaled_weights, 0)
            pygame.pixelcopy.array_to_surface(
                self.weights_surf[i],
                red * 0x010000 + green * 0x000100,
            )

        # Update colored gene surface
        self.colored_gene_surf.fill(self.color)
        layer_height = self.HEIGHT // len(self.weights_surf)
        for i, weight_surf in enumerate(self.weights_surf):
            self.colored_gene_surf.blit(
                pygame.transform.scale(
                    weight_surf, (self.WIDTH, layer_height)
                ),
                (0, i * layer_height),
            )

        super().reset_state(track)

    def draw(self, screen: pygame.Surface, camera: Camera):
        """
        Draw the AI car.

        :param screen: The screen
        :param camera: The camera
        :return: None
        """
        rotated_surf = pygame.transform.rotate(
            self.colored_gene_surf.convert_alpha(),
            np.degrees(camera.rot - self.rot),
        )

        if not self.out_of_track:
            super().draw_sensor(screen, camera)

        coord = camera.get_coord(self.pos) - vec(*rotated_surf.get_size()) // 2
        screen.blit(rotated_surf, coord)
