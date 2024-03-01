# Python Experiment Helpers

The main feature of this helper is a framework that allows defining experiments
and can run them in parallel, taking care of all stuff around it, mainly:

- Experiments have a clear structure relating experiment labels
  to functions and input parameters.
- Experiments can be structured like a tree, and their results are stored in
  a corresponding tree of directories to make them easy to browse.
- Experiments can be run in parallel with no coding overhead.
  This includes running them on SLURM to use GPUs.
- Each function can always just use `logging` for updates and
  save results to the current working directory, and the framework make sure
  this always works out correctly.
- Experiments that have results are not run twice by default (can be forced).
- Constants can be configured with configuration objects that work well with
  IDEs and can be updated at runtime.
- Logging is set up automatically and includes optional file and slack logging.


**Open Issues**

- [ ] Document the CLI a bit better, in particular config and slurm flags.
- [ ] Remove checking for any existing files; we have the metafile now.
      As a consequence, we can also remove config for tmp files.
- [ ] `-l` and `-d` are redundant, we could settle for one of the two.
- [ ] Make universal `write` and `read` methods that infer from file ending.


## Installation

```bash
pip install git+ssh://git@gitlab.ethz.ch/nsg/employees/adietmue/projects/python-experiment-helpers.git#egg=experiment_helpers
```

Then, you should be able to import the helpers in python:

```python
import experiment_helpers as eh
```


## Using the framework

For a working example, see the example [run.py](run.py).
The most important features are documented below.


### Organize experiments and run via CLI

Each experiment is simply defined by a function, and you only need to create
a dictionary of experiment names with their corresponding functions.
The helpers give you an easy CLI to run functions organized in this way.

In a nutshell, create a file `run.py` and register your experiments:

```python
import experiment_helpers as eh
from my_module import some_experiment_function, another_experiment_function

# Create a group of experiments called `main` from a dictionary of experiments.
main_experiments = eh.framework.ParametrizedExperiments("main", {
    'experiment_1': some_experiment_function,
    'experiment_2': another_experiment_function,
})

if __name__ == "__main__":
    # Start the CLI
    eh.cli.experiment_cli()  # pylint: disable=no-value-for-parameter
```

Then you can use the CLI to run experiments:

```bash
python run.py --help             # show groups of experiments
python run.py main --help        # show all experiments within `main`
python run.py main experiment_1  # specific matching
python run.py main "*_1"         # wildcard matching
python run.py main "*_1" "*_2"   # multiple patterns
python run.py main --jobs=2      # run 2 experiments in parallel
python run.py main --slurm       # run the experiments on SLURM
```

### Experiment functions

What kind of functions can be used for experiments? Basically any!

```python
def some_experiment():
    """A simple experiment."""
    ...

experiments = {
    "some_experiment": some_experiment,
}
```

Often multiple experiments use the same function, but with different inputs.
You can easily parametrize experiments using [functools.partial](https://docs.python.org/3/library/functools.html#functools.partial)
from the Python standard library:

```python
from functools import partial

def some_experiment(*, param):
    """A simple experiment with a parameter."""
    ...

experiments = {
    "experiment_1": partial(some_experiment, param=41),
    "experiment_42": partial(some_experiment, param=42),
}
```

Once you have defined a group of experiments, create a `ParametrizedExperiments`
instance, which will be used to start the experiments.
The group needs a name, and the experiments will be automatically available
in the CLI under that name. But you can also use the instance to start
experiments from within Python.

```python
import experiment_helpers as eh

main_experiments = eh.framework.ParametrizedExperiments("main", experiments)

# Run experiments from within Python. For the CLI, see above.
main_experiments()
```

### CLI and multiprocessing

Finally, somewhere you need a script that calls `eh.cli.experiment_cli()`,
which will start the CLI. Let's call this file `run.py`:

```python
import experiment_helpers as eh

# Import your code such that the CLI functions get registered.
# import ...

if __name__ == "__main__":
    # Start the CLI
    eh.cli.experiment_cli()  # pylint: disable=no-value-for-parameter
```
As you can see, this file does not need to do anything itself, but it is
important that it imports all the code that defines the experiments;
if `ParametrizedExperiments` is not called, the CLI will not know about it!

Now you can run the CLI using `python run.py`, or you can make it executable
with `chmod +x run.py` (only once) and go for an even shorter: `./run.py`.
This will run all experiments, but you can also use the `--help` flag to
see all available options, or match only specific experiments:

```bash
python run.py --help
python run.py main --help
python run.py main experiment_1  # specific matching
python run.py main "*_42"        # wildcard matching
python run.py main "*_42" "*_1"  # multiple patterns
python run.py main --jobs=2      # run 2 experiments in parallel
python run.py main --slurm       # run experiments on SLURM
```

As you can see, using the `--jobs` or `-j` flag, you can start multiple
experiments in parallel. Similarly, using `--slurm`, you can force scheduling
the experiments on SLURM (make sure that SLURM is available).
If your experiment function uses multiple processes itself, you should use the
Configuration helpers (see below) and use the `config.workers` attribute
to set the number of processes.
Using the CLI, you can then control the number of experiment processes using
the `--workers` or `-w` flag. For example, `python run.py main -j2 -w2` will
start two experiments in parallel, and set `config.workers` to `2` for each
of them, which you can use in your functions.


### Logging

If you want to keep track of whats going on, use the standard `logging` module:

```python
import logging

def some_experiment():
    """A simple experiment that logs its status."""
    logging.critical("This is a critical log.")
    logging.error("This is an error log.")
    logging.warning("This is an warning log.")
    logging.info("This is an info log.")
    logging.debug("This is an debug log.")
```

Normally, thes does not work well with multiprocessing, and in particular
not with distributing jobs across SLURM nodes.
The framework takes care of these issues, so you can always rely on the
`logging` module to work as expected, even when running experiments in parallel
or on SLURM.

When using the CLI, you can control the loglevel using the `-v` flag.

```bash
python run.py main      # critical, error, warning.
python run.py main -v   # critical, error, warning, info.
python run.py main -vv  # critical, error, warning, info, debug.
```

Logging also supports [sending messages to Slack](#slack-integration).

### Configure experiments

In a large project, it can be hard to keep track of constants and default
values. In addition, you may need to change this config based on the machine
you are running on (e.g. to set the correct filesystem paths) or even
override some defaults for specific experiments.
Enter `eh.config.BaseConfig`, a configuration base class with some
useful features to address these issues.

Why a class and not a dictionary? Because most IDEs can autocomplete class
attributes, and you can easily see all available attributes in one place.
This helps to avoid typos and makes looking up attributes easier.
Additionally, classes support dynamic properties and inheritance, all making
it easier to define and use configs.

The core functionality is provided by the `with_updates` class method, which
allows to use a specific config class for defaults, and update it with more
specific values

First, define your config with additional attributes you need,
inheriting from `eh.config.BaseConfig`. Then, you can use it in your functions
like this:


```python
import experiment_helpers as eh

class CustomConfig(eh.config.BaseConfig):
    """A user config class."""

    # Add some user attributes.
    user_message: str = "Hello world!"
    user_number: int = 42

    # Machine specific attributes, e.g. to always use SLURM.
    slurm = True

def withconfig(*, config = None):
    """A simple function with config."""
    config = CustomConfig.with_updates(config)

    # In addition to the general updates,
    # you can also override specific attributes:
    # overrides = {'user_message': "Goodbye world!"}
    # config = CustomConfig.with_updates(config, overrides)
```

What is happening here? `config` will always be a `CustomConfig` instance,
thus all attributes are available. If a `config` is provided, it will be used
to update the defaults.
You can call the function like this, optionally with config updates:

```python
withconfig()
withconfig(config={"user_message": "Hello world!", "logger": "somelogger"})
```

The framework also uses the config. If you want to change the config defaults
used by the framework and CLI, provide them when instantiating the `ParametrizedExperiments` object:

```python
main_experiments = eh.framework.ParametrizedExperiments(
    "main", experiments, configcls=CustomConfig)
```

In particular, this is useful to change the base `output_directory` or logging
settings.

Finally, the framework makes it easy to use a machine-specific config in
addition to the default config. If there is a file `config.py` which contains
a class `Config`, it will be used to update the default config automatically.
You can also use the `--config` flag to specify a different file or class.

```bash
python run.py main                                          # load default
python run.py main --config some_config.py                  # file
python run.py main --config some_config.py:ConfigClassName  # file and class
```

Take a look at the [BaseConfig class](experiment_helpers/config.py#L89)
to see the attributes used by the framework. You can modify them to suit your
needs.

### Structuring experiments

The name of the experiment is used to create a directory for outputs as a
subdirectory of the `output_dir` in the config (which defaults to `./results`).

You can use a path-like name to order experiments in subdirectories:

```python
experiments = {
    "some/subdir/experiment_a": some_experiment,
    "some/subdir/experiment_b": some_experiment,
}
```

The only limitation is that all experiments must be leaves of the directory
tree such that all experiment files can be kept separate. That means the
following is not allowed:

```python
experiments = {
    "some/subdir/experiment_a": some_experiment,
    "some/subdir/experiment_b": some_experiment,
    "some/subdir": some_experiment,  # not a leaf!
}
```
Not that experiment names may not start or end with a `/`.

### Checkpoints and temporary files

The framework considers an experiment done when it created any files.
As you may want to create some temporary files or checkpoints before your
experiment has finished, any experiment containing files with the word
`checkpoint` is always continued, and otherwise, files ending in `.tmp` are
ignored. You can add additional patterns by modifying the `checkpoint_files`
and `ignore_files` patterns in the config.

If the experiment is run with the `--force` flag, all files in the directory
are deleted, including ignore files. The experiment will start fresh.

#### Slack Integration

To use the Slack integration, you need to set up a Slack app and add it to
a channel you want to use for messages.

Go to the [slack website](https://api.slack.com/apps/) and create an app for
your workspace.
In the homepage of your newly created app, click on `OAuth & Permissions` in
the sidebar and ensure it has the `chat:write` scope under _Bot Token Scopes_.
Then, copy the _Bot User OAuth Token_.

Second, you need the id of the channel to use for slack messages.
Find the channel URL by right clicking on your channel and selecting
_Copy link_, which will result in a URL like
`https://workspace.slack.com/archives/XYZ`, where `XYZ` is the channel id.

Finally, update your [config](#configure-experiments):

```python
class Config(eh.config.BaseConfig):
    """A user config class."""

    # All your other config

    slacktoken = "the token for your app"
    slackchannel = "the channel id"
```

Afterwards, all logs are automatically sent to slack, including summary
messages containing info whether experiments succeeded or failed.

### SSH and SLURM

TODO: Document slurm options?

### Typing

`experiment_helpers.typing` contains all standard `typing` types, numpy
`ArrayLike` as well as some useful custom types like paths and config types.
Helps to reduce import clutter.


## Development

Use `pip install -e .` to install the package in editable mode.

Tests are run using `pytest`.

### Varia

Follow
[these instructions](https://click.palletsprojects.com/en/8.0.x/setuptools/)
to install the script above as an entrypoint.
Instead of `cli`, use `eh.cli.experiment_cli`.
