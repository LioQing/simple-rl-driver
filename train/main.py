import argparse
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import pygame

from engine.entity.camera import Camera
from engine.entity.car import AICar
from engine.entity.track import Track

DESCRIPTION = (
    "Training mode.\n"
    "\n"
    "controls:\n"
    "  ctrl + q                        quit the program\n"
    "  ctrl + s                        save the weights\n"
    "     enter                        manually trigger next iteration\n"
)

WEIGHTS_PATH = Path("data/weights")


def main_scene(args: argparse.Namespace):
    """
    Main scene for training the AI
    :return: None
    """
    pygame.init()
    pygame.display.set_caption("Simple RL Driver - Training")
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(
        args.resolution,
        (pygame.FULLSCREEN if args.fullscreen else 0) | pygame.RESIZABLE,
    )

    # Setup
    track = Track.load(args.track)

    ai_cars = [AICar() for _ in range(args.ai_count)]

    # Weight
    weights_exist = True
    weights_file = WEIGHTS_PATH / f"{args.weights}.txt"
    if not weights_file.exists():
        weights_file.touch()
        weights_exist = False

    weights = weights_file.read_text().splitlines()

    if not weights:
        weights_exist = False

    for i, car in enumerate(ai_cars):
        car.set_start_pos(track)
        if weights_exist:
            car.nn.from_string(weights[i % len(weights)])
            if i > len(weights):
                car.nn.mutate(0.01)

    camera = Camera(screen)

    # Next iteration function for the AI
    def next_iteration():
        select_count = 2
        ai_cars.sort(key=lambda x: x.get_fitness(), reverse=True)
        for i in range(select_count, len(ai_cars)):
            ai_cars[i].nn = deepcopy(ai_cars[i % select_count].nn)

            # Change if needed
            ai_cars[i].nn.mutate(0.2, 0.5, ai_cars[i].get_fitness())

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
                    and event.key == pygame.K_q
                ):
                    running = False
                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_s
                ):
                    weights_file.write_text(
                        "\n".join(str(car.nn) for car in ai_cars)
                    )
                if event.key == pygame.K_RETURN:
                    next_iteration()
                    for car in ai_cars:
                        car.set_start_pos(track)

        # If all cars out of track then next iteration
        if all(car.out_of_track for car in ai_cars):
            next_iteration()
            for car in ai_cars:
                car.set_start_pos(track)

        # Update
        for car in ai_cars:
            if not car.out_of_track:
                car.update(fixed_dt, track)

        camera.car = max(ai_cars, key=lambda x: x.get_fitness())
        camera.update(fixed_dt)

        # Draw
        screen.fill((255, 255, 255))

        track.draw(screen, pygame.Color(0, 0, 0), camera, 5)
        for car in ai_cars:
            car.draw(screen, pygame.Color(0, 0, 0), camera)

        pygame.display.update()

        if args.limit_fps:
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
        "--weights",
        "-w",
        dest="weights",
        type=str,
        help="The weights file to use for the AI",
        required=True,
    )
    parser.add_argument(
        "--limit-fps",
        "-l",
        dest="limit_fps",
        action="store_true",
        help="Whether to run with limited 60 fps",
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
        "--ai-count",
        "-a",
        dest="ai_count",
        type=int,
        help="The number of AI cars to use",
        default=10,
    )


def main():
    """
    Entry point for the program in training mode
    :return: None
    """
    parser = argparse.ArgumentParser(
        description="Entry point for the program in training mode"
    )
    configure_parser(parser)
    args = parser.parse_args()
    main_scene(args)


if __name__ == "__main__":
    main()
