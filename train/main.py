import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

from engine.activations import activation_funcs
from engine.entity.ai_car import AICar

DESCRIPTION = (
    "Training mode.\n"
    "\n"
    "controls:\n"
    "     enter                        manually trigger next iteration\n"
    "  ctrl + s                        save the neural network\n"
    "  ctrl + q                        quit the program\n"
)

NN_DIR = Path("data/nns")


def load_nn(
    args: argparse.Namespace,
) -> Tuple[
    List[float],
    Optional[List[str]],
    Optional[List[int]],
    str,
    List[int],
]:
    """
    Load the neural network from the file.

    ```
    | Return value       | NN loaded | NN not found |
    | ------------------ | --------- | ------------ |
    | Sensor rotations   | Yes       | Yes          |
    | Weights            | Yes       | No           |
    | Hidden layer sizes | No        | Yes          |
    | Activation         | Yes       | Yes          |
    | Color              | Yes       | Yes          |
    ```

    :param args: The arguments
    :return: The return values
    """
    nn_file = NN_DIR / f"{args.nn}.txt"
    if not nn_file.exists():
        if args.sensor_rots is None or args.hidden_layer_sizes is None:
            raise ValueError(
                "Sensor rotations (--sensor-rot/-s) and hidden layer sizes"
                " (option --hidden-layer-sizes/-z) must be provided when the"
                " neural network file does not exist"
            )
        sensor_rots = [math.radians(r) for r in args.sensor_rots]
        return (
            sensor_rots,
            None,
            args.hidden_layer_sizes,
            args.activation,
            args.color,
        )
    else:
        meta, *weights = nn_file.read_text().splitlines()
        sensor_rots_str, activation, color_str = meta.split(";")
        sensor_rots = [float(r) for r in sensor_rots_str.split(",")]
        color = tuple(int(c) for c in color_str.split(","))
        return sensor_rots, weights, None, activation, color


def save_nn(args: argparse.Namespace, ai_cars: List[AICar]):
    """
    Save the neural network to the file

    :param args: The arguments
    :param ai_cars: The AI cars
    :return: None
    """
    nn_file = NN_DIR / f"{args.nn}.txt"
    nn_file.touch()

    sensor_rots = ",".join(str(rot) for rot in ai_cars[0].sensor_rots)
    color = ",".join(str(c) for c in ai_cars[0].color)
    activation = list(activation_funcs.keys())[
        list(activation_funcs.values()).index(ai_cars[0].nn.activation)
    ]
    weights = "\n".join(
        car.nn.serialize()
        for car in ai_cars[: args.save_quota or len(ai_cars)]
    )
    nn_file.write_text(f"{sensor_rots};{activation};{color}\n{weights}")


def main_scene(args: argparse.Namespace):
    """
    Main scene for training the AI

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
        "--sensor-rots",
        "--sensor-rot",
        "-s",
        dest="sensor_rots",
        type=float,
        nargs="+",
        help=(
            "The sensor rotations for the AI cars, in degrees, space"
            " separated. Required if neural network file is not found, it will"
            " be used to create a new neural network"
        ),
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        "-z",
        dest="hidden_layer_sizes",
        type=int,
        nargs="+",
        help=(
            "The hidden layer sizes for the neural network. Required if neural"
            " network file is not found, it will be used to create a new"
            " neural network"
        ),
    )
    parser.add_argument(
        "--activation-function",
        "-f",
        dest="activation",
        type=str,
        choices=activation_funcs.keys(),
        help="The activation function for the neural network",
        default="leaky_relu",
    )
    parser.add_argument(
        "--save-quota",
        "-q",
        dest="save_quota",
        type=int,
        help=(
            "The number of quota of top AI cars to save into the neural"
            " network file"
        ),
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
        "--select-count",
        "-c",
        dest="select_count",
        type=int,
        help="The number of top AI cars to select for next iteration",
        default=3,
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
        "--mutate-noise",
        "-m",
        dest="mutate_noise",
        type=float,
        help="The mutation noise (scale of Gaussian distribution)",
        default=0.2,
    )
    parser.add_argument(
        "--mutate-learn-rate",
        "-l",
        dest="mutate_learn_rate",
        type=float,
        help="The mutation learn rate (magnitude of gradient descent)",
        default=0.5,
    )
    parser.add_argument(
        "--color",
        "-r",
        dest="color",
        type=int,
        nargs=3,
        help="The color of the AI car",
        default=(0, 0, 0),
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
        "--limit-fps",
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
