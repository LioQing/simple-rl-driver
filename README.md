# Simple RL Driver

A reinforcement learning project for training AI cars to drive on tracks.

## Table of Contents

- [Overview](#overview)
    - [Libraries](#libraries)
    - [Features](#features)
- [Examples](#examples)
    - [Training](#training)
    - [Gameplay](#gameplay)
- [Usage](#usage)
    - [Track Editor Mode](#track-editor-mode)
    - [Training Mode](#training-mode)
    - [Gameplay Mode](#gameplay-mode)
- [Environment Setup](#environment-setup)
    - [Python](#python)
    - [Virtual Environment](#virtual-environment)
    - [Lint and Pre-commit](#lint-and-pre-commit)
    - [VS Code Settings](#vs-code-settings)
- [Development](#development)
    - [Clone Repository](#clone-repository)
    - [Checkout Branch](#checkout-branch)
    - [Committing Changes](#committing-changes)
    - [Pushing Changes](#pushing-changes)

## Overview

This project try to be as simple as possible, using only basic libraries and tools to demonstrate the concept and mathematical foundation of reinforcement learning.

### Libraries

- [NumPy](https://numpy.org): For numerical operations and linear algebra.
- [Pygame](https://www.pygame.org): For game development.
- [Shapely](https://shapely.readthedocs.io): For geometric operations, specifically ray casting and collision detection.

### Features

- Track Editor: Create and edit tracks with curves.
    - Create tracks using Bezier curves, written in Python from scratch.
    - Save and load tracks from files.
- Training Mode: Train AI cars to drive on tracks using neural networks and genetic algorithms.
    - Adjust hyperparameters such as mutation noise, learn rate, and hidden layer sizes.
    - Save and load neural networks from files.
    - Visualize weights and biases of the neural network directly on the car.
- Gameplay Mode: Play the game with the AI cars.
    - Follow the AI cars or control the player car.

## Examples

In the [`data`](./data) directory, there are some demo files for neural networks and tracks.

### Training

Run the following command to train the AI cars on the tracks in [`data/tracks/`](./data/tracks/).

```bash
python main.py train -t demo0 demo1 demo2 -n my-nn -s -90 -45 0 45 90 -z 4 4 2 -f leaky_relu -q 3 -a 10 -c 3
```

Options:

- `-t demo0 demo1 demo2`: Use the tracks `demo0`, `demo1`, and `demo2`.
    - Tracks are always loaded from the [`data/tracks/`](./data/tracks/) directory.
    - The tracks are selected at random for each iteration.
- `-n my-nn`: Use the neural network file `my-nn` to save the weights and biases.
    - If the file exists, the weights and biases are loaded from the file, and the AI cars are initialized with them, try using `demo`.
    - If the file does not exist, a new neural network is created with the specified hyperparameters.
    - If you want to overwrite existing files, you need to delete them manually.
    - Press `Ctrl + S` to save the neural network, you should be able to see the file in the [`data/nns/`](./data/nns/) directory.
- `-s -90 -45 0 45 90`: Use the sensor rotations `-90`, `-45`, `0`, `45`, and `90` degrees.
    - The sensor rotations are the direction of which the AI cars can sense the track's edges.
- `-z 4 4 2`: Use the hidden layer sizes `4 4 2`.
    - The hidden layer sizes are the number of neurons in each hidden layer.
    - The order of the hidden layer sizes is from the input layer to the output layer, excluding the input and output layers.
- `-f leaky_relu`: Use the activation function `leaky_relu`.
    - If unspecified, the default activation function is `leaky_relu`.
- `-q 3`: Save the top 3 AI cars into the neural network file.
- `-a 10`: Use 10 AI cars.
- `-c 3`: Select the top 3 AI cars for the next iteration.

More tips:

- You can manually trigger the next iteration by pressing `Space`.
- You can adjust the initial mutation noise, mutation noise, and mutation learn rate using `-i`, `-m`, and `-l`.

For more options, see the [Training Mode](#training-mode) section or run `python main.py train -h`.

### Gameplay

Run the following command to play the game with the AI cars using the neural network [`data/nns/demo.txt`](./data/nns/demo.txt) on the track [`data/tracks/demo0.txt`](./data/tracks/demo0.txt).

```bash
python main.py game -t demo0 -n demo -a 5
```

Options:

- `-t demo0`: Use the track `demo0`.
- `-n demo`: Use the neural network file `demo` to load the weights and biases.
- `-a 5`: Use 5 AI cars.

More tips:

- You choose to not use a player car and follow the best AI cars using `--follow-ai`.
- You can adjust the initial mutation noise using `-i`.

For more options, see the [Gameplay Mode](#gameplay-mode) section or run `python main.py game -h`.

## Usage

```bash
usage: main.py [-h] {game,track,train} ...
```

### Track Editor Mode

Use this mode to edit tracks.

```bash
usage: main.py track [-h]
                     --track TRACK
                     [--resolution RESOLUTION RESOLUTION]
                     [--fullscreen]
```

Controls:

| Key | Action |
| :---: | :---: |
| Press left mouse button | Add a point |
| Hold and drag left mouse button | Edit the curve |
| Press right mouse button | Remove a point |
| Press `Ctrl` + `S` | Save the track |
| Press `Ctrl` + `Z`, `Del`, `Backspace`, `Esc` | Undo |
| Press `Ctrl` + `Q` | Quit the program |

Options:

| Short | Long | Default | Required | Type | Help | Remark |  
| :---: | :---: | :---: | :---: | :--: | :---: | :---: |  
| `-h` | `--help` | | | `bool` | Show this help message and exit. | |  
| `-t` | `--track` | | Yes | `str` | The name of the track to edit. | |  
| | `--resolution` | `800 640` | No | `tuple[int, int]` | The resolution of the track. | |  
| | `--fullscreen` | | No | `bool` | Whether to run in fullscreen mode. | |

### Training Mode

Use this mode to train AI car neural networks.

```bash
usage: main.py train [-h]
                     --track TRACKS [TRACKS ...]
                     --neural-network NN
                     [--sensor-rots SENSOR_ROTS [SENSOR_ROTS ...]]
                     [--hidden-layer-sizes HIDDEN_LAYER_SIZES [HIDDEN_LAYER_SIZES ...]]
                     [--activation-function {softmax,sigmoid,relu,leaky_relu}]
                     [--save-quota SAVE_QUOTA]
                     [--ai-count AI_COUNT]
                     [--select-count SELECT_COUNT]
                     [--init-mutate-noise INIT_MUTATE_NOISE]
                     [--mutate-noise MUTATE_NOISE]
                     [--mutate-learn-rate MUTATE_LEARN_RATE]
                     [--color COLOR COLOR COLOR]
                     [--color-gene]
                     [--limit-fps]
                     [--skip-frames SKIP_FRAMES]
                     [--resolution RESOLUTION RESOLUTION]
                     [--fullscreen]
```

Controls:

| Key | Action |
| :---: | :---: |
| Press `Enter` | Manually trigger next iteration |
| Press `Ctrl` + `S` | Save the neural network |
| Press `Ctrl` + `Q` | Quit the program |

Options:

| Short | Long | Default | Required | Type | Help | Remark |  
| :---: | :---: | :---: | :---: | :--: | :---: | :---: |  
| `-h` | `--help` | | | `bool` | Show this help message and exit. | |  
| `-t` | `--track`, `--tracks` | | Yes | `list[str]` | The name of the track(s) to train in. | |  
| `-n` | `--neural-network` | | Yes | `str` | The neural network file to use for the AI. | |  
| `-s` | `--sensor-rots`, `--sensor-rot` | | No | `list[float]` | The sensor rotations for the AI cars, in degrees, space separated. | Required if neural network file is not found, it will be used to create a new neural network. |  
| `-z` | `--hidden-layer-sizes` | | No | `list[int]` | The hidden layer sizes for the neural network. | Required if neural network file is not found, it will be used to create a new neural network. |  
| `-f` | `--activation-function` | `leaky_relu` | No | `str` | The activation function for the neural network. | Must be one of `["softmax", "sigmoid", "relu", "leaky_relu"]`. |  
| `-q` | `--save-quota` | | No | `int` | The number of quota of top AI cars to save into the neural network file. | |  
| `-a` | `--ai-count` | `10` | No | `int` | The number of AI cars to use. | |  
| `-c` | `--select-count` | `3` | No | `int` | The number of top AI cars to select for next iteration. | |  
| `-i` | `--init-mutate-noise` | `0.01` | No | `float` | The initial mutation noise (scale of Gaussian distribution). | Only used if weights are loaded from neural network file and `ai-count` is greater than the number of weights |  
| `-m` | `--mutate-noise` | `0.2` | No | `float` | The mutation noise (scale of Gaussian distribution). | |  
| `-l` | `--mutate-learn-rate` | `0.5` | No | `float` | The mutation learn rate (magnitude of gradient descent). | |  
| `-r` | `--color` | `0 0 0` | No | `tuple[int, int, int]` | The color of the AI car. | Only used if neural network file is not found. |  
| | `--color-gene`, `--colored-gene` | | No | `bool` | Whether to use colored gene for the AI car. | Overrides the color of the AI cars. |  
| | `--limit-fps` | | No | `bool` | Whether to run with limited 60 fps. | |  
| | `--skip-frames` | `0` | No | `int` | The number of frames to skip for each update. | Enabling this disables the 60 fps limit. |  
| | `--resolution` | `800 640` | No | `tuple[int, int]` | The resolution of the track. | |  
| | `--fullscreen` | | No | `bool` | Whether to run in fullscreen mode. | |

### Gameplay Mode

Use this mode to play the game with the AI.

```bash
usage: main.py game [-h]
                    --track TRACK
                    [--neural-network NN]
                    [--ai-count AI_COUNT]
                    [--init-mutate-noise INIT_MUTATE_NOISE]
                    [--color COLOR COLOR COLOR]
                    [--follow-ai]
                    [--color-gene]
                    [--resolution RESOLUTION RESOLUTION]
                    [--fullscreen]
```

Controls:

| Key | Action |
| :---: | :---: |
| Press `Ctrl` + `R` | Restart the program |
| Press `Ctrl` + `Q` | Quit the program |

Options:

| Short | Long | Default | Required | Type | Help | Remark |  
| :---: | :---: | :---: | :---: | :--: | :---: | :---: |  
| `-h` | `--help` | | | `bool` | Show this help message and exit. | |  
| `-t` | `--track` | | Yes | `str` | The name of the track to play in. | |  
| `-n` | `--neural-network` | | No | `str` | The neural network file to use for the AI. | |  
| `-a` | `--ai-count` | `10` | No | `int` | The number of AI cars to use. | |  
| `-i` | `--init-mutate-noise` | `0.01` | No | `float` | The initial mutation noise (scale of Gaussian distribution). | |  
| `-r` | `--color` | `0 0 0` | No | `tuple[int, int, int]` | The color of the player car. | |  
| | `--follow-ai` | | No | `bool` | Whether to follow the AI car and disable player car. | |  
| | `--color-gene`, `--colored-gene` | | No | `bool` | Whether to use colored gene for the AI cars. | Overrides the color of the AI cars. |  
| | `--resolution` | `800 640` | No | `tuple[int, int]` | The resolution of the track. | |  
| | `--fullscreen` | | No | `bool` | Whether to run in fullscreen mode. | |

## Environment Setup

### Python

We use Python 3.9.13, so make sure you have that installed.

You could use [pyenv](https://github.com/pyenv/pyenv) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (Windows is not recommended to install pyenv because it does not get native support) to manage your Python versions.

Install the Python version you want to use.
```bash
pyenv install 3.9.13
```

Specify the version for this directory.
```bash
pyenv local 3.9.13
```

To check your Python version, run `python --version` in your terminal.
```bash
python --version
```
Or you may need to specify the version explicitly if you didn't use pyenv or have multiple versions installed.
```bash
python3 --version
```

### Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

It is highly recommended to use the [venv](https://docs.python.org/3/library/venv.html) module that comes with Python.

To create a virtual environment in the `.venv` directory, run:
```bash
python -m venv .venv
```

Activate the environment.
```bash
# Linux, Bash, Mac OS X
source .venv/bin/activate
# Linux, Fish
source .venv/bin/activate.fish
# Linux, Csh
source .venv/bin/activate.csh
# Linux, PowerShell Core
.venv/bin/Activate.ps1
# Windows, cmd.exe
.venv\Scripts\activate.bat
# Windows, PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies.
```bash
pip install -r requirements.txt
```

When you want to deactivate the virtual environment.
```bash
deactivate
```

### Lint and Pre-commit

We use [Flake8](https://flake8.pycqa.org) and [ISort](https://pycqa.github.io/isort/) for the coding style and guidelines. The style is then enforced by [pre-commit](https://pre-commit.com).

Finish the environment setup above (especially installing the dependencies with pip) before using pre-commit.

Install and setup pre-commit.
```bash
pre-commit install
```

To run pre-commit manually (only scans staged files).
```bash
pre-commit run --all-files
```

Remember to stage files again if there are any changes made by the pre-commit hooks or by you.
```bash
git add .
```

### VS Code Settings

You can add a workspace setting to automatically format your code on save using the black formatter.

You need to have the [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) VS Code extension installed.

Bring up the command palette with Ctrl+Shift+P(Windows/Linux) / Cmd+Shift+P(Mac) and search for "Preferences: Open Workspace Settings (JSON)".

Then replace the content with the following:
```json
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
    },
    "black-formatter.args": [
        "--line-length",
        "79",
        "--preview",
        "--enable-unstable-feature",
        "string_processing"
    ],
}
```

## Development

### Clone Repository

First clone the repository.
```bash
git clone git@github.com:<username>/<repository>.git
```

**Important**: You may need to setup SSH keys for your GitHub account. See [this guide](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) for more information.

### Checkout Branch

Then checkout the branch you want to work on.
```bash
git checkout <branch>
```

### Committing Changes

Commit your changes to the branch you are working on.
```bash
git add .
git commit -m "Your commit message"
```

Make any changes and stage your files again according to the pre-commit hooks.

### Pushing Changes

Set your branch's upstream branch to be the same branch on the remote repository on GitHub.
```bash
git push -u origin <branch>
```

After the first time you set the upstream branch, you can simply push without specifying the branch.
```bash
git push
```
