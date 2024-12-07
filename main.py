import argparse

import game.main
import track.main
import train.main

DESCRIPTION = "Main entry point for the program"


def configure_parser(parser: argparse.ArgumentParser):
    """
    Configure the parser for the program
    :param parser: The parser to configure
    :return: None
    """
    # Add a subparser to the main parser
    # This allows for mode-specific parsers to parse arguments related to only
    # that mode
    subparsers = parser.add_subparsers(
        dest="mode",
        help="The mode to run in",
        required=True,
    )

    # Add game parser
    game_parser = subparsers.add_parser(
        "game",
        help="Run in gameplay mode",
        description=game.main.DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
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

    # Add train parser
    train_parser = subparsers.add_parser(
        "train",
        help="Run in training mode",
        description=train.main.DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train.main.configure_parser(train_parser)


def main():
    """
    Main entry point for the program
    :return: None
    """
    # Create the argument parser with the description
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    # Configure the main parser
    configure_parser(parser)

    # Parse the arguments
    args = parser.parse_args()

    # Run the main_scene with the arguments based on the `mode` argument
    if args.mode == "game":
        game.main.main_scene(args)
    elif args.mode == "track":
        track.main.main_scene(args)
    elif args.mode == "train":
        train.main.main_scene(args)


if __name__ == "__main__":
    main()
