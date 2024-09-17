import argparse
from typing import Tuple

import pygame

DESCRIPTION = (
    "Game play mode.\n"
    "\n"
    "controls: to be determined\n"
)


def main_scene(args: argparse.Namespace):
    """
    Main scene for playing the game
    :return: None
    """
    pygame.init()
    pygame.display.set_caption("Simple RL Driver - Game Play")
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(
        args.resolution,
        (pygame.FULLSCREEN if args.fullscreen else 0) | pygame.RESIZABLE,
    )


def configure_parser(parser: argparse.ArgumentParser):
    """
    Configure the parser for the program
    :param parser: The parser to configure
    :return: None
    """
    parser.add_argument(
        "--track", "-t",
        dest="track",
        type=str,
        help="The name of the track to play in",
        required=True,
    )
    parser.add_argument(
        "--resolution", "-r",
        dest="resolution",
        type=Tuple[int, int],
        help="The resolution of the track",
        default=(800, 640),
    )
    parser.add_argument(
        "--fullscreen", "-f",
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
