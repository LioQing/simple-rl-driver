import argparse

import game.main
import track.main

DESCRIPTION = "Main entry point for the program"


def configure_parser(parser: argparse.ArgumentParser):
    """
    Configure the parser for the program
    :param parser: The parser to configure
    :return: None
    """
    subparsers = parser.add_subparsers(
        dest="mode",
        help="The mode to run in",
        required=True,
    )

    # Add game parser
    game_parser = subparsers.add_parser(
        "game",
        help="Run in game mode",
    )
    game.main.configure_parser(game_parser)

    # Add track parser
    track_parser = subparsers.add_parser(
        "track",
        help="Run in track editor mode",
        description=track.main.DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    track.main.configure_parser(track_parser)


def main():
    """
    Main entry point for the program
    :return: None
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    configure_parser(parser)
    args = parser.parse_args()

    if args.mode == "game":
        raise NotImplementedError("Game mode is not implemented")
    elif args.mode == "track":
        track.main.main_scene(args)


if __name__ == "__main__":
    main()
