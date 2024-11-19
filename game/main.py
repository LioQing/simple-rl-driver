import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pygame

from engine.activations import activation_funcs
from engine.entity.ai_car import AICar
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


def load_nn(args: argparse.Namespace) -> Tuple[List[float], List[str], str]:
    """
    Load the neural network from the file

    :param args: The arguments
    :return: The sensor rotations, weights, and the activation
    """
    nn_file = NN_PATH / f"{args.nn}.txt"
    if not nn_file.exists():
        raise FileNotFoundError(f"Weights file {nn_file} does not exist")

    meta, *weights = nn_file.read_text().splitlines()
    if len(weights) == 0:
        raise ValueError(f"Weights file {nn_file} is empty")

    sensor_rots_str, activation = meta.split(";")

    sensor_rots = [float(r) for r in sensor_rots_str.split(",")]

    return sensor_rots, weights, activation


def main_scene(args: argparse.Namespace):
    """
    Main scene for playing the game

    :param args: The arguments
    :return: None
    """
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
        sensor_rots, weights, activation = load_nn(args)

        ai_cars = [
            AICar(
                track,
                np.array(sensor_rots, dtype=np.float32),
                weights=weights[i % len(weights)],
                init_mutate_noise=args.init_mutate_noise,
                activation=activation_funcs[activation],
            )
            for i in range(args.ai_count)
        ]

    camera = Camera(screen, player_car)

    # Restart function
    def restart():
        if not args.follow_ai:
            player_car.reset_state(track)

        for car in ai_cars:
            car.reset_state(track)

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

        camera.update(
            fixed_dt,
            (
                max(ai_cars, key=lambda x: x.fitness)
                if args.follow_ai
                else None
            ),
        )

        # draws
        screen.fill(pygame.Color(255, 255, 255))

        track.draw(screen, camera, 5)

        if not args.follow_ai:
            player_car.draw(screen, camera)

        for car in ai_cars:
            car.draw(screen, camera)

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
