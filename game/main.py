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
    # Initialize pygame by calling `init`
    pygame.init()

    # Set the window title with `set_caption`
    pygame.display.set_caption("Simple RL Driver - Game Play")

    # Create a clock object to help control the frame rate
    clock = pygame.time.Clock()

    # Create a screen with `set_mode`
    #
    # `args.resolution` is a tuple[int, int] in the form of (width, height)
    #
    # `args.fullscreen` is a boolean indicating whether to run in fullscreen
    # mode
    screen = pygame.display.set_mode(
        args.resolution,
        (pygame.FULLSCREEN if args.fullscreen else 0) | pygame.RESIZABLE,
    )

    # Setup the track by loading it
    #
    # `args.track` is a string representing the name of the track to play in
    track = Track.load(args.track)

    # Create a player car object
    #
    # `args.color` is a tuple[int, int, int] representing the color of the
    # player car
    player_car = PlayerCar(color=pygame.Color(*args.color))

    # If `args.follow_ai` is False, reset the state of the player car by
    # calling `reset_state` with the track
    if not args.follow_ai:
        player_car.reset_state(track)

    # Create a list of AI cars
    ai_cars = []

    # If `args.nn` is provided, load the neural network arguments using
    # `load_nn`
    if args.nn:
        sensor_rots, weights, activation, color = load_nn(args)

        # Create a list of AI cars
        #
        # `args.ai_count` is the number of AI cars
        #
        # Get the activation function using `activation_funcs[activation]`
        #
        # Take the i-th element of `weights` if it is not None, otherwise None
        #
        # Supply `init_mutate_noise` with `args.init_mutate_noise`
        ai_cars = [
            AICar(
                np.array(sensor_rots, dtype=np.float32),
                weights=weights[i % len(weights)],
                init_mutate_noise=args.init_mutate_noise,
                activation=activation_funcs[activation],
                color=pygame.Color(*color),
            )
            for i in range(args.ai_count)
        ]

        # Reset the state of each AI car by calling `reset_state` with the
        # track
        for car in ai_cars:
            car.reset_state(track)

    # Create a camera object
    #
    # If `args.follow_ai` is True, follow the first AI car, otherwise follow
    # the player car
    camera = Camera(screen, player_car)

    # Define the restart function
    def restart():
        # If `args.follow_ai` is False, reset the state of the player car by
        # calling `reset_state` with the track
        if not args.follow_ai:
            player_car.reset_state(track)

        # Reset the state of each AI car by calling `reset_state` with the
        # track
        for car in ai_cars:
            car.reset_state(track)

    # Main loop forever while `running` is True
    running = True
    fixed_dt = 0.032
    while running:
        # Handle events from `pygame.event.get()`
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If it is a quit event, we stop the loop
                running = False
            elif event.type == pygame.KEYDOWN:
                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_r
                ):
                    # If control + r is pressed, we restart the program
                    restart()
                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_q
                ):
                    # If control + q is pressed, we quit the program
                    running = False

        # Update the player car if `args.follow_ai` is False
        if not args.follow_ai:
            player_car.update(fixed_dt, track)

        # Update each AI car
        #
        # Skip the car if it is out of track
        for car in ai_cars:
            if car.out_of_track:
                continue

            car.update(fixed_dt, track)

        # If AI follow mode is enabled, sort
        # the AI cars by fitness in descending order
        if args.follow_ai:
            ai_cars.sort(key=lambda x: x.fitness, reverse=True)

        # Update the camera to follow the first AI car if `args.follow_ai` is
        # True, otherwise follow the player car
        camera.update(fixed_dt, args.follow_ai and ai_cars[0])

        # Clear the screen with white color by using `fill` method on the
        # screen object
        screen.fill(pygame.Color(255, 255, 255))

        # Draw the track and the cars on the screen
        track.draw(screen, camera, 5)

        # Draw the player car if `args.follow_ai` is False
        if not args.follow_ai:
            player_car.draw(screen, camera)

        # Draw each AI car on the screen
        for car in ai_cars:
            car.draw(screen, camera)

        # Update the display with `update`
        pygame.display.update()

        # Tick the clock to control the frame rate
        clock.tick(60)

    # Quit pygame with `quit`
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
