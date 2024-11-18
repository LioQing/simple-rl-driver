# Python Template

This is a template for Python projects.

## Usage

```bash
usage: main.py [-h] {game,track,train} ...
```

### Track Editor Mode

Use this mode to edit tracks.

```bash
usage: main.py track [-h]
                     --track TRACK
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
| | `--resolution` | `(800, 640)` | No | `tuple[int, int]` | The resolution of the track. | |  
| | `--fullscreen` | | No | `bool` | Whether to run in fullscreen mode. | |

### Training Mode

Use this mode to train AI car neural networks.

```bash
usage: main.py train [-h]
                     --track TRACKS [TRACKS ...] --neural-network NN
                     [--sensor-rots SENSOR_ROTS [SENSOR_ROTS ...]]
                     [--hidden-layer-sizes HIDDEN_LAYER_SIZES [HIDDEN_LAYER_SIZES ...]]
                     [--activation-function {sigmoid,relu,leaky_relu}]
                     [--save-quota SAVE_QUOTA]  
                     [--ai-count AI_COUNT]
                     [--select-count SELECT_COUNT]  
                     [--init-mutate-noise INIT_MUTATE_NOISE]
                     [--mutate-noise MUTATE_NOISE]  
                     [--mutate-learn-rate MUTATE_LEARN_RATE]
                     [--limit-fps]
                     [--skip-frames SKIP_FRAMES]  
                     [--resolution RESOLUTION]  
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
| `-t` | `--track`, `--tracks` | | Yes | `list[str]` | The name of the track(s) to train in. | Must specify at least one track. |  
| `-n` | `--neural-network` | | Yes | `str` | The neural network file to use for the AI. | |  
| `-s` | `--sensor-rots`, `--sensor-rot` | | No | `list[float]` | The sensor rotations for the AI cars, in degrees, space separated. | Required if neural network file is not found, it will be used to create a new neural network. |  
| `-z` | `--hidden-layer-sizes` | | No | `list[int]` | The hidden layer sizes for the neural network. | Required if neural network file is not found, it will be used to create a new neural network. |  
| `-f` | `--activation-function` | `leaky_relu` | No | `str` | The activation function for the neural network. | Must be one of `["sigmoid", "relu", "leaky_relu"]`. |  
| `-q` | `--save-quota` | | No | `int` | The number of quota of top AI cars to save into the neural network file. | |  
| `-a` | `--ai-count` | `10` | No | `int` | The number of AI cars to use. | |  
| `-c` | `--select-count` | `3` | No | `int` | The number of top AI cars to select for next iteration. | |  
| `-i` | `--init-mutate-noise` | `0.01` | No | `float` | The initial mutation noise (scale of Gaussian distribution). | |  
| `-m` | `--mutate-noise` | `0.2` | No | `float` | The mutation noise (scale of Gaussian distribution). | |  
| `-l` | `--mutate-learn-rate` | `0.5` | No | `float` | The mutation learn rate (magnitude of gradient descent). | |  
| | `--limit-fps` | | No | `bool` | Whether to run with limited 60 fps. | |  
| | `--skip-frames` | `0` | No | `int` | The number of frames to skip for each update. Enabling this also disables the 60 fps limit. | |  
| | `--resolution` | `(800, 640)` | No | `tuple[int, int]` | The resolution of the track. | |  
| | `--fullscreen` | | No | `bool` | Whether to run in fullscreen mode. | |

### Gameplay Mode

Use this mode to play the game with the AI.

```bash
usage: main.py game [-h]
                    --track TRACK
                    [--neural-network NN]
                    [--ai-count AI_COUNT]
                    [--init-mutate-noise INIT_MUTATE_NOISE]
                    [--follow-ai]  
                    [--resolution RESOLUTION]
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
| | `--follow-ai` | | No | `bool` | Whether to follow the AI car and disable player car. | |  
| | `--resolution` | `(800, 640)` | No | `tuple[int, int]` | The resolution of the track. | |  
| | `--fullscreen` | | No | `bool` | Whether to run in fullscreen mode. | |

## Environment Setup

### Python

We use Python \<version>, so make sure you have that installed.

You could use [pyenv](https://github.com/pyenv/pyenv) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (Windows is not recommended to install pyenv because it does not get native support) to manage your Python versions.

Install the Python version you want to use.
```bash
pyenv install <version>
```

Specify the version for this directory.
```bash
pyenv local <version>
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
