import argparse
from pathlib import Path
from typing import Tuple

import pygame

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

WEIGHTS_PATH = Path("data/weights")


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
    player_car = PlayerCar()
    if not args.follow_ai:
        player_car.reset_state(track)

    ai_cars = []
    if args.weights:
        ai_cars = [AICar() for _ in range(args.ai_count)]

        # Weight
        weights_file = WEIGHTS_PATH / f"{args.weights}.txt"
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Weights file {weights_file} does not exist"
            )

        weights = weights_file.read_text().splitlines()
        for i, car in enumerate(ai_cars):
            car.reset_state(track)
            car.nn.from_string(weights[i % len(weights)])
            if i > len(weights):
                car.nn.mutate(0.01)

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
                max(ai_cars, key=lambda x: x.get_fitness())
                if args.follow_ai
                else None
            ),
        )

        # draws
        screen.fill((255, 255, 255))

        track.draw(screen, pygame.Color(0, 0, 0), camera, 5)

        if not args.follow_ai:
            player_car.draw(screen, pygame.Color(0, 0, 0), camera)

        for car in ai_cars:
            car.draw(screen, pygame.Color(0, 0, 0), camera)

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
        "--resolution",
        "-r",
        dest="resolution",
        type=Tuple[int, int],
        help="The resolution of the track",
        default=(800, 640),
    )
    parser.add_argument(
        "--fullscreen",
        "-f",
        dest="fullscreen",
        action="store_true",
        help="Whether to run in fullscreen mode",
    )
    parser.add_argument(
        "--weights",
        "-w",
        dest="weights",
        type=str,
        help="The weights file to use for the AI",
    )
    parser.add_argument(
        "--follow-ai",
        "-n",
        dest="follow_ai",
        action="store_true",
        help="Whether to follow the AI car and disable player car",
    )
    parser.add_argument(
        "--ai-count",
        "-a",
        dest="ai_count",
        type=int,
        help="The number of AI cars to use",
        default=10,
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
