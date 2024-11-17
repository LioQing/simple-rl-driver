import argparse
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import pygame

from engine.entity.ai_car import AICar
from engine.entity.camera import Camera
from engine.entity.track import Track

DESCRIPTION = (
    "Training mode.\n"
    "\n"
    "controls:\n"
    "  ctrl + q                        quit the program\n"
    "  ctrl + s                        save the neural network\n"
    "     enter                        manually trigger next iteration\n"
)

NN_DIR = Path("data/nns")


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
    tracks = [Track.load(t) for t in args.tracks]
    track = random.choice(tracks)
    camera = Camera(screen)

    nn_file = NN_DIR / f"{args.nn}.txt"
    if not nn_file.exists():
        weights = []
        sensor_rots = [math.radians(r) for r in args.sensor_rots]
        if sensor_rots is None:
            raise ValueError(
                "Sensor rotations (option -s) are required if no neural"
                " network file"
            )
    else:
        meta, *weights = nn_file.read_text().splitlines()
        sensor_rots = [float(r) for r in meta.split(",")]

    ai_cars = [
        AICar(np.array(sensor_rots, dtype=np.float32))
        for _ in range(args.ai_count)
    ]

    for i, car in enumerate(ai_cars):
        car.reset_state(track)
        if weights:
            car.nn.from_str(weights[i % len(weights)])
            if i > len(weights):
                car.nn.mutate(0.01)

    # Next iteration function for the AI
    def next_iteration():
        select_count = 2
        ai_cars.sort(key=lambda x: x.fitness, reverse=True)

        for i in range(select_count, len(ai_cars)):
            ai_cars[i].nn = deepcopy(ai_cars[i % select_count].nn)
            ai_cars[i].nn.mutate(0.2, 0.5, ai_cars[i].fitness)

        for car in ai_cars:
            car.reset_state(track)

    # Main loop
    running = True
    fixed_dt = 0.032
    skip_frame_counter = 0
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
                    nn_file.touch()
                    nn_file.write_text(
                        ",".join(str(rot) for rot in ai_cars[0].sensor_rots)
                        + "\n"
                        + "\n".join(str(car.nn) for car in ai_cars)
                    )
                if event.key == pygame.K_RETURN:
                    track = random.choice(tracks)
                    next_iteration()

        # If all cars out of track then next iteration
        if all(car.out_of_track for car in ai_cars):
            track = random.choice(tracks)
            next_iteration()

        # Update
        for car in ai_cars:
            if not car.out_of_track:
                car.update(fixed_dt, track)

        ai_cars = sorted(ai_cars, key=lambda x: x.fitness, reverse=True)
        camera.update(fixed_dt, ai_cars[0])

        # Skip frames
        skip_frame_counter += 1
        if skip_frame_counter < args.skip_frames:
            continue
        skip_frame_counter = 0

        # Render
        screen.fill((255, 255, 255))

        track.draw(screen, pygame.Color(0, 0, 0), camera, 5)
        for car in ai_cars:
            car.draw(screen, pygame.Color(0, 0, 0), camera)

        pygame.display.update()

        if args.limit_fps and args.skip_frames == 0:
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
        "--tracks",
        "-t",
        dest="tracks",
        type=str,
        nargs="+",
        help="The name of the track(s) to train in",
        required=True,
    )
    parser.add_argument(
        "--neural-network",
        "-n",
        dest="nn",
        type=str,
        help="The neural network file to use for the AI",
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
        "--skip-frames",
        dest="skip_frames",
        type=int,
        help=(
            "The number of frames to skip for each update, enabling this also"
            " disables the 60 fps limit"
        ),
        default=0,
    )
    parser.add_argument(
        "--resolution",
        dest="resolution",
        type=Tuple[int, int],
        help="The resolution of the track",
        default=(800, 640),
    )
    parser.add_argument(
        "--fullscreen",
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
    parser.add_argument(
        "--sensor-rots",
        "--sensor-rot",
        "-s",
        dest="sensor_rots",
        type=float,
        nargs="+",
        help=(
            "The sensor rotations for the AI cars, in degrees, space"
            " separated, it is required if no neural network file is specified"
        ),
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
