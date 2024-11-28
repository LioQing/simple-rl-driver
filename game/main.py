import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pygame

from engine.activations import activation_funcs
from engine.car_nn_vis import CarNNVis
from engine.entity.ai_car import AICar
from engine.entity.ai_colored_gene_car import AIColoredGeneCar
from engine.entity.camera import Camera
from engine.entity.player_car import PlayerCar
from engine.entity.track import Track

DESCRIPTION = (
    "Game play mode.\n"
    "\n"
    "controls:\n"
    "  ctrl + r                        restart the program\n"
    "  ctrl + q                        quit the program\n"
)

NN_PATH = Path("data/nns")


def load_nn(
    args: argparse.Namespace,
) -> Tuple[List[float], List[str], str, List[int]]:
    """
    Load the neural network from the file

    :param args: The arguments
    :return: The sensor rotations, weights, activation, color
    """
    nn_file = NN_PATH / f"{args.nn}.txt"
    if not nn_file.exists():
        raise FileNotFoundError(f"Weights file {nn_file} does not exist")

    meta, *weights = nn_file.read_text().splitlines()
    if len(weights) == 0:
        raise ValueError(f"Weights file {nn_file} is empty")

    sensor_rots_str, activation, color_str = meta.split(";")

    sensor_rots = [float(r) for r in sensor_rots_str.split(",")]
    color = tuple(int(c) for c in color_str.split(","))

    return sensor_rots, weights, activation, color


def main_scene(args: argparse.Namespace):
    """
    Main scene for playing the game

    :param args: The arguments
    :return: None
    """
    # Argument check
    if args.follow_ai and not args.nn:
        raise ValueError(
            "AI follow mode `--follow-ai` requires neural network"
            " `--neural-network`"
        )

    if args.nn_vis and not args.nn:
        raise ValueError(
            "Neural network visualization `--nn-vis` requires neural network"
            " `--neural-network`"
        )

    # Pygame setup
    pygame.init()
    pygame.display.set_caption("Simple RL Driver - Game Play")
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(
        args.resolution,
        (pygame.FULLSCREEN if args.fullscreen else 0) | pygame.RESIZABLE,
    )

    # Setup
    track = Track.load(args.track)
    player_car = PlayerCar(color=pygame.Color(*args.color))
    if not args.follow_ai:
        player_car.reset_state(track)

    ai_cars = []
    if args.nn:
        sensor_rots, weights, activation, color = load_nn(args)
        ai_car_cls = AIColoredGeneCar if args.color_gene else AICar

        ai_cars = [
            ai_car_cls(
                np.array(sensor_rots, dtype=np.float32),
                weights=weights[i % len(weights)],
                init_mutate_noise=args.init_mutate_noise,
                activation=activation_funcs[activation],
                color=pygame.Color(*color),
            )
            for i in range(args.ai_count)
        ]

        for car in ai_cars:
            car.reset_state(track)

    if args.nn_vis:
        car_nn_vis = CarNNVis(
            args.nn_vis, ai_cars[0].nn.layer_sizes, activation
        )
        car_nn_vis.set_weights(ai_cars[0].nn.weights)

    camera = Camera(screen, player_car)

    # Restart function
    def restart():
        if not args.follow_ai:
            player_car.reset_state(track)

        for car in ai_cars:
            car.reset_state(track)

        if args.nn_vis:
            car_nn_vis.set_weights(ai_cars[0].nn.weights)

    # Main loop
    running = True
    fixed_dt = 0.032
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_r
                ):
                    restart()
                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_q
                ):
                    running = False

        # Update
        if not args.follow_ai:
            player_car.update(fixed_dt, track)

        for car in ai_cars:
            if car.out_of_track:
                continue

            car.update(fixed_dt, track)

        if args.nn_vis:
            prev_first_car = ai_cars[0]

        if args.nn_vis or args.follow_ai:
            ai_cars.sort(key=lambda x: x.fitness, reverse=True)

        camera.update(fixed_dt, args.follow_ai and ai_cars[0])

        if args.nn_vis:
            if id(prev_first_car) != id(ai_cars[0]):
                car_nn_vis.set_weights(ai_cars[0].nn.weights)

            car_nn_vis.set_nodes(
                ai_cars[0].inputs,
                ai_cars[0].nn.hiddens,
                ai_cars[0].outputs,
            )

        # Draws
        screen.fill(pygame.Color(255, 255, 255))

        track.draw(screen, camera, 5)

        if not args.follow_ai:
            player_car.draw(screen, camera)

        for car in ai_cars:
            car.draw(screen, camera)

        if args.nn_vis:
            car_nn_vis.draw(screen, (0, 0))

        pygame.display.update()

        clock.tick(60)

    pygame.quit()


def configure_parser(parser: argparse.ArgumentParser):
    """
    Configure the parser for the program

    :param parser: The parser to configure
    :return: None
    """
    parser.add_argument(
        "--track",
        "-t",
        dest="track",
        type=str,
        help="The name of the track to play in",
        required=True,
    )
    parser.add_argument(
        "--neural-network",
        "-n",
        dest="nn",
        type=str,
        help="The neural network file to use for the AI",
    )
    parser.add_argument(
        "--ai-count",
        "-a",
        dest="ai_count",
        type=int,
        help="The number of AI cars to use",
        default=10,
    )
    parser.add_argument(
        "--init-mutate-noise",
        "-i",
        dest="init_mutate_noise",
        type=float,
        help="The initial mutation noise (scale of Gaussian distribution)",
        default=0.01,
    )
    parser.add_argument(
        "--color",
        "-r",
        dest="color",
        type=int,
        nargs=3,
        help="The color of the player car",
        default=(0, 0, 0),
    )
    parser.add_argument(
        "--follow-ai",
        dest="follow_ai",
        action="store_true",
        help="Whether to follow the AI car and disable player car",
    )
    parser.add_argument(
        "--color-gene",
        "--colored-gene",
        dest="color_gene",
        action="store_true",
        help="Whether to use colored gene for the AI cars",
    )
    parser.add_argument(
        "--nn-vis",
        "--neural-network-visual",
        dest="nn_vis",
        type=int,
        nargs=2,
        help="The size of the neural network visualization",
    )
    parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        nargs=2,
        help="The resolution of the track",
        default=(800, 640),
    )
    parser.add_argument(
        "--fullscreen",
        dest="fullscreen",
        action="store_true",
        help="Whether to run in fullscreen mode",
    )


def main():
    """
    Entry point for the program in game mode

    :return: None
    """
    parser = argparse.ArgumentParser(
        description="Entry point for the program in game mode"
    )
    configure_parser(parser)
    args = parser.parse_args()
    main_scene(args)


if __name__ == "__main__":
    main()
