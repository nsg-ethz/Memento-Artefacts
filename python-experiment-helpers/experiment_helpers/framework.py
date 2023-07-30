"""A utility framework to run experiments.

For details, check out the the README, but see a full usage example below:

```
import logging
from functools import partial

from experiment_helpers.framework import (BaseConfig, ParametrizedExperiments,
                                          experiment_cli)

# Define a default config for your experiments.
class ExperimentConfig(BaseConfig):
    important_attribute = 42
    output_directory = 'experiment_results/'

# Define experiment functions. They can accept a config argument for config
# update at runtime, e.g. machine-specific settings.
def experiment_function(config = None, kwarg = None):
    config = BaseConfig.with_updates(config)
    logging.info(attribute)
    logging.info(kwarg)
    # Write results simply to current directory.

# Define parametrized experiments
experiments = ParametrizedExperiments(
    "important",  # Name (all outputs are under this directory)
    {
        'run1/exp1': partial(experiment_function, kwarg=1),
        'run1/exp2': partial(experiment_function, kwarg=2)
    },
    configcls=ExperimentConfig
)

# Run. You can update config at runtime.
# The following example provides a specific output directory.

# Run everything with the default config.
experiments()

# Run everything with a config updates.
experiments(config={'output_directory': 'ichangedmymind/'})

# Run selected experiments (UNIX patterns are allowed)
experiments("run1/*", config={'output_directory': 'ichangedmymind/'})


# Conretely, the functions are executed from the directories
# `experiment_results/important/run1/exp1` and
# `experiment_results/important/run1/exp2` respectively. They can simply save
# data to the current directory and everything will end up in the right place.

# Experiments are automatically added to a CLI, which can be run with:
experiment_cli()
```
"""

import fnmatch
import inspect
import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime as dt
from itertools import chain
from os import chdir
from textwrap import dedent

import click

from . import executor, log_utils
from .config import BaseConfig, load_config_from_pyfile
from .data import Path, to_json
from .typing import (Any, Callable, Dict, Literal, Optional, Sequence, Type,
                     Union, cast)

# CLI base group.
# ===============


@click.group()
def experiment_cli():
    """Run experiments."""


# Parametrization.
# ================

class ParametrizedExperiments:
    """Define a parametrized set of experiments.

    Returns a functions that allows to easily run these parametrized
    experiments in parallel, with helpful features:
    - each experiment is run in it's own output directory.
    - if the output already exists, skip experiments unless `force` is True.
    - experiments can be run in parallel.

    The experiment functions must accept a configuration object.
    The experiment key doubles as the path where the function is run
    relative to `config.output_directory`. In the following example, `function`
    is run from `/tmp/set1/experiment1`, and if it saves any data, it will end
    up there.

    ```
    class Config(BaseConfig):
        output_directory = '/tmp'

    experiments = {
        'set1/experiment1`: function,
    }

    parametrized = parametrize(experiments, Config)
    parametrized()
    ```

    Keys must not start with leading or trailing `/`.

    See docs of __call__ for detailed usage once parametrized.
    """

    def __init__(
        self,
        name,
        experiments: Dict[str, Callable],
        configcls: Type[BaseConfig] = BaseConfig,
        cli: Optional[Union[Literal[False], callable]] = experiment_cli,
        cli_help: Optional[str] = None,
    ):
        # Checks to make sure things go smooth.
        for experiment in experiments:
            assert not experiment.startswith('/'), \
                "Experiment labels must not start with `/`."
            assert not experiment.endswith('/'), \
                "Experiment labels must not end with `/`."
        assert_all_leaves(experiments)

        assert issubclass(configcls, BaseConfig), \
            "Config must inherit from BaseConfig."

        self.name = name
        self.experiments = experiments
        self.configcls = configcls

        if cli:
            add_to_cli(self, cli=cli, cli_help=cli_help)

    def available_experiments(self, patterns: Union[str, Sequence[str]] = "*"):
        """List available experiments (matching patterns)."""
        return self.get_matches(patterns if patterns else "*")

    def get_matches(self, patterns: Union[str, Sequence[str]] = "*"):
        """Return all experiment keys matching patterns."""
        return list(chain.from_iterable(
            _matches(self.experiments, pattern) for pattern in patterns
        ))

    def __call__(self,
                 patterns: Union[str, Sequence[str]] = "*",
                 config: Optional[Any] = None, n_jobs: int = 1,
                 force: bool = False, dry=False,
                 ):
        """Run all experiments matched by patterns.

        Matching uses UNIX-like patterns, e.g. `*` matches everything.
        You can provide a single string to match or multiple matches.

        Functions will be run in directories equal to their label as a subdir
        of `config.output_directory`, e.g. if the output directory is `/tmp`,
        and the experiment is `test1/exp1`, it will be run from the directory
        `/tmp/test1/exp` and can safely store output in the current directory.

        If the directory contains any files already, the experiment will be
        skipped _unless_ `force` is True.

        You can provide user-config-overrides that will be applied to all
        experiments.

        You can run multiple jobs in parallel by specifying n_jobs.

        Specific `dry=True` only logs which experiments will be run/skipped, but
        does not do anything.
        """
        # Load own config for output, logging and as base for experiment config.
        config = self.configcls.with_updates(config)
        outdir = Path(config.output_directory) / self.name
        outdir.mkdir(exist_ok=True, parents=True)
        with log_utils.server_logging(self.name, patterns, config) \
                as runlogger:
            # Match experiments and check output.
            runlogger.info("Experiment group: `%s`", self.name)
            runlogger.debug("Matching experiments.")
            patterns = [patterns] if isinstance(patterns, str) else patterns
            matches = self.get_matches(patterns)
            assert matches, f"No experiment matches the pattern `{patterns}`!"

            # Prepare jobs to run including config overrides.
            runlogger.debug("Preparing jobs.")
            running, skipped, jobargs = [], [], []
            for match in matches:
                joboutdir = outdir / match
                if not force and _is_completed(joboutdir, config):
                    skipped.append(match)
                    continue
                running.append(match)

                # Pickle args so the job runs with multiprocessing, SLURM, etc.
                # Use cloudpickle instead of pickle to pass all kinds of values
                # that pickle cannot handle, e.g. lambdas and dynamic classes.
                jobargs.append(
                    (match, self.experiments[match], joboutdir, config)
                )

            if skipped:
                runlogger.debug(dedent("""
                Skipping the following %i experiments:
                ```
                %s
                ```
                """).strip(), len(skipped), "\n".join(skipped))

            if running:
                runlogger.info(dedent("""
                Running the following %i experiments:
                ```
                %s
                ```
                """).strip(), len(running), "\n".join(running))
            else:
                runlogger.info("Nothing to do.")
                return

            if dry:
                return

            if force:
                # If force, we don't skip anything. Ensure that no old files
                # are left that may cause problems with the new run.
                runlogger.info(
                    "Forcing fresh execution, removing previous files.")
                for match in matches:
                    _remove_files(outdir / match)

            # Run jobs.
            start = dt.utcnow()
            job_results = executor.map(_run, jobargs,
                                       config=config, logger=runlogger)
            results = list(zip(running, job_results))
            duration = dt.utcnow() - start

            # Report results.
            successes = [match for match, success in results if success]
            failures = [match for match, success in results if not success]

            # Wait for job logs before reporting summary.
            runlogger.info(
                "All jobs ended after %s. Waiting for logs.", duration)
            log_utils.wait_for_logs(config, jobargs)

            if successes and failures:
                runlogger.warning("Not all experiments were successful.")
            elif not successes and failures:
                runlogger.error("All experiments failed.")
            else:
                runlogger.info("All experiments completed successully.")

            # Details.
            if failures:
                runlogger.error(dedent("""
                The following %i experiments crashed:
                ```
                %s
                ```
                """).strip(), len(failures), "\n".join(failures))


# Helper functions.
# =================

def add_to_cli(parametrized: ParametrizedExperiments,
               cli=experiment_cli, cli_help: Optional[str] = None):
    """Add parametrized experiments as a CLI command."""
    @cli.command(name=parametrized.name)
    @click.option('-l', '--list', 'list_exps', is_flag=True,
                  help="List available experiments.")
    @click.option('-d', '--dry', 'dry', is_flag=True,
                  help="Dry run, don't do any work, only show preparation.")
    @click.option('-c', '--config',  # default="config.py",  # see callback!
                  type=click.Path(readable=True, dir_okay=False),
                  callback=_config_callback,
                  help="Python file that contains a `Config` class.")
    @click.option('-v', '--verbose', 'loglevel', count=True,
                  callback=_verbosity_callback,
                  help="Increase logging verbosity.")
    @click.option('-f', '--force', is_flag=True,
                  help="Force execution, even if results exist.")
    @click.option('-j', '--jobs', 'n_jobs', type=click.INT,
                  help="Number of jobs to run in parallel.")
    @click.option('-w', '--workers', type=click.INT,
                  help="Max number of worker processes _per job_.")
    @click.option('--slurm/--no-slurm', default=None,
                  help="Force or prevent SLURM scheduling.")
    @click.argument("patterns", nargs=-1)
    def _run_parametrized(patterns, config=None, force: bool = False,
                          list_exps=False, loglevel: int = logging.WARNING,
                          n_jobs: Optional[int] = None,
                          workers: Optional[int] = None,
                          slurm: Optional[bool] = None,
                          dry: bool = False):
        """Run experiments."""
        if list_exps:
            exps = parametrized.available_experiments(patterns)
            click.echo(f"{len(exps)} available experiments:")
            for exp in exps:
                click.echo(exp)
            return

        # Create userconfig with overrides from CLI options.
        overrides = dict(
            loglevel=min(loglevel, getattr(config, "loglevel", loglevel)),
        )
        if workers is not None:
            overrides['workers'] = workers
        if n_jobs is not None:
            overrides['multiprocessing_jobs'] = n_jobs
            overrides['slurm_simultaneous_jobs'] = n_jobs
        if slurm is not None:
            overrides['slurm'] = slurm
        userconfig = parametrized.configcls.with_updates(config, overrides)

        if not patterns:
            patterns = "*"
        return parametrized(patterns, config=userconfig, force=force, dry=dry)

    if cli_help is not None:
        _run_parametrized.help = cli_help


def _matches(experiments, pattern):
    """Return experiments matching pattern.

    If nothing matches, try to see if there are sub experimeents, and if so,
    return those.
    """
    matches = [key for key in experiments if fnmatch.fnmatch(key, pattern)]
    if not (matches or pattern.endswith("*")):
        # Try sub-experiments.
        if pattern.endswith("/"):  # Avoid double slash.
            pattern = pattern[:-1]
        return _matches(experiments, f"{pattern}/*")
    return matches


def _run(args):
    """Run a function with loggin, working directory, and config set up."""
    label, function, outdir, config = args
    config = cast(BaseConfig, config)  # For mypy.
    # Change dir (create it if needed) and run logging queue.
    with _change_dir(outdir), log_utils.client_logging(label, config):
        logging.info("Starting.")
        start = dt.utcnow()
        try:
            # Run function with config if it takes a config argument.
            if "config" in inspect.signature(function).parameters:
                function(config=config)
            else:
                function()
            end = dt.utcnow()
            duration = end - start
            # Write a file with metadata about the completed experiment.
            # This allows to infer wheter it was completed or not.
            to_json({
                "start": start.strftime(config.utcfmt),
                "end": end.strftime(config.utcfmt),
                "seconds": duration.total_seconds(),
                # TODO: We could add a hash of the config here or sth to detect
                #       changes to the config that may invalidate the results?
            }, outdir / "experiment_helpers_metadata.json")
            logging.info("Completed in %s.", duration)
            return True
        except:  # pylint: disable=bare-except
            logging.exception("Error!")
            return False


@contextmanager
def _change_dir(path: Path):
    """Change cwd to path and back."""
    missing_dirs = []  # Create, and remove again later (if still empty).
    current = Path(path)
    while not current.exists():
        missing_dirs.append(current)
        if current == current.parent:
            break  # we are at the root.
        current = current.parent

    Path(path).mkdir(parents=True, exist_ok=True)  # Create everything.
    current = Path('.').absolute()
    try:
        chdir(path)
        yield
    finally:
        chdir(current)
        # Clean up dirs that were created earlier (if no output).
        for _dir in missing_dirs:
            try:
                _dir.rmdir()
            except OSError:
                break  # Files were created, no need to remove.


def assert_all_leaves(experiments):
    """Return true if all experiments are leaves."""
    children = defaultdict(list)
    # First run: add each experiment as children to all parents in tree.
    for exp in experiments:
        current_path = Path(exp)
        while current_path.parent != current_path:
            children[str(current_path.parent)].append(exp)
            current_path = current_path.parent
    # Second run: check that all experiments are leaves.
    for exp in experiments:
        sub_exps = children[exp]
        assert not sub_exps, \
            f"`{exp}` is not a leaf. Conflicts: `{sub_exps}`."


def _is_completed(path: Path, config: BaseConfig):
    """Return True if the path contains any (non-ignored) files."""
    if not path.is_dir():
        return False

    for item in path.iterdir():
        is_ignored = any(fnmatch.fnmatch(item.name, pattern)
                         for pattern in config.ignore_files)
        is_checkpoint = any(fnmatch.fnmatch(item.name, pattern)
                            for pattern in config.checkpoint_files)
        if item.is_file():
            if is_checkpoint:
                return False  # checkpoints - > not completed
            if not is_ignored:
                return True  # a non-ignored file -> completed
        elif item.is_dir() and not is_ignored:
            # If it's a non-ignored dict, check recursively.
            if _is_completed(item, config=config):
                return True
    return False


def _remove_files(path: Path):
    """Remove all files from path, if any."""
    if not path.is_dir():
        return  # Nothing to do.
    for item in path.iterdir():
        if item.is_file():
            item.unlink()  # Clear file.
        elif item.is_dir():
            _remove_files(item)  # Clear dir.


# For CLI callbacks, lets ignore that we need to specify some callback args.
# pylint: disable=unused-argument

def _config_callback(ctx, param, value):
    """Load config file."""
    default_classname = 'Config'
    if value is None:  # No config file provided, try default.
        default_config = Path("./config.py")
        if default_config.exists():  # Default exists, use it.
            filename, key = default_config, default_classname
        else:  # Skip if no value provided and default does not exist.
            return None
    elif ':' in value:  # Both filename and classname provided.
        filename, key = value.split(':')
    else:  # Only filename provided, use default classname.
        filename, key = value, default_classname

    click.echo(f"Loading `{key}` from `{filename}`.")
    return load_config_from_pyfile(filename, key=key)


def _verbosity_callback(ctx, param, value):
    """Map the number of `v` to verbosity."""
    return {  # Map count to logging level.
        # Default to warnings and errors.
        0: logging.WARNING,
        1: logging.INFO,
    }.get(value, logging.DEBUG)  # verbosity >= 2.

# pylint: enable=unused-argument
