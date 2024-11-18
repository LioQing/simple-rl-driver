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
    subparsers = parser.add_subparsers(
        dest="mode",
        help="The mode to run in",
        required=True,
    )

    # Add game parser
    game_parser = subparsers.add_parser(
        "game",
        help="Run in gameplay mode",
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
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    configure_parser(parser)
    args = parser.parse_args()

    if args.mode == "game":
        game.main.main_scene(args)
    elif args.mode == "track":
        track.main.main_scene(args)
    elif args.mode == "train":
        train.main.main_scene(args)


if __name__ == "__main__":
    main()
