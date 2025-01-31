import argparse
from pathlib import Path
from typing import List, Tuple

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
    pass


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
