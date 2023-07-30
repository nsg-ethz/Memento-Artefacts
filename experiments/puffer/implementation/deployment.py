"""Prepare model for deployment to the puffer platform.

Concretely, puffer expects five independent models (can be parallelized);
one for each step into the horizon (0, 1, .., 4 steps into the future).
Each model has a corresponding index, i.e. index 0 is horizon 0.

For each day and index, we store the following files:
- `/{day}/stats_{index}.pickle.gz` contains statistics (see code below).
- `/{day}/data_{index}.pickle.gz` memory content after data selection.
- (if retraining needed) model files in `/{day}/model`.
"""

import contextlib
import fcntl
import logging
import shutil
import subprocess
from os import PathLike
from experiment_helpers.data import Path
from typing import Union

import experiment_helpers as eh
import numpy as np
import pandas as pd

from memento.utils import merge_datadicts

from ..config import PufferDeploymentConfig
from . import memory, models
from .data import load_inout

PathTypeype = Union[PathLike, str]
MemType = type[memory.PufferMemory]


class NoDataError(FileNotFoundError):
    """Missing inout data."""


memlog = logging.getLogger("memory")

# `*` syntax means: all arguments are keyword arguments!


def update(*, config=None,
           first_day, last_day,
           memcls: MemType, index: int, threshold: float,
           git_dir=None):
    """Update the deployment model."""
    config = PufferDeploymentConfig.with_updates(config)
    first_day = pd.to_datetime(first_day)
    last_day = pd.to_datetime(last_day)
    second_last_day = last_day - pd.Timedelta(days=1)  # For better logging.

    # Load checkpoint.
    checkpointfile = Path(".").absolute() / "checkpoint.pickle"
    if checkpointfile.is_file():
        logging.debug("Loading checkpoint data from `%s`.", checkpointfile)
        last_valid_stats = eh.data.read_pickle(checkpointfile)
        # Adjust starting day based on checkpoint
        first_day = last_valid_stats['day'] + pd.Timedelta(days=1)
    else:
        logging.debug("No previous checkpoint.")
        last_valid_stats = {}

    # And go!
    for day in pd.date_range(first_day, last_day, freq='D'):
        daystr = config.daystr(day)
        memlog.info("Day: `%s` ", daystr)
        if day.date() == last_day.date():
            logging.info("Day: `%s`", daystr)
        else:
            logging.debug("Day: `%s`", daystr)

        try:
            # Process day. If everything works, update last valid stats.
            last_valid_stats = _update_day(
                day=day, last_stats=last_valid_stats,
                memcls=memcls, index=index, threshold=threshold,
                config=config,
            )
            # Update checkpoint.
            logging.debug("Saving checkpoint to `%s`.", checkpointfile)
            eh.data.to_pickle(last_valid_stats, checkpointfile)
        except NoDataError as error:
            msg = f"Day `{daystr}` has no data."
            if day.date() == last_day.date():
                logging.critical(
                    "Data for current day missing! You need to download it.")
                raise error
            if day.date() == second_last_day.date():  # More important.
                logging.warning(msg)
            else:
                logging.debug(msg)
        except Exception as error:  # pylint: disable=broad-except
            msg = f"Day `{daystr}` crashed!"
            if day.date() == last_day.date():
                logging.critical("Current day crashed!")
                raise error
            if day.date() == second_last_day.date():  # More important.
                logging.warning(msg)
            else:
                logging.debug(msg)

    logging.debug("Memory is up-to-date.")

    model_dir = last_valid_stats["model_dir"]
    if (git_dir is not None) and (model_dir is not None):
        logging.debug("Synchronizing git.")
        _sync_git(
            index=index, logger=logging,
            model_dir=model_dir, git_dir=git_dir,
        )
    elif git_dir is None:
        logging.debug("Git disabled.")
    else:
        logging.debug("Git enabled, but no model to push.")


def _update_day(*, config=None,
                day, last_stats: dict,
                memcls: MemType, index: int, threshold: float) -> dict:
    """Compute updates for a specific day and return path to stat file.

    The statistics file contains information on which data was used for
    the memory, training, as well as the results from coverage analysis.

    The last statfile (if any) is needed to correctly initialize the memory
    based on the last valid day.
    """
    day = pd.to_datetime(day)
    data_day = day - pd.Timedelta(days=1)  # On day x, data from x-1 is avail.
    config = PufferDeploymentConfig.with_updates(config)

    daydir = Path(".").absolute() / config.daystr(day) / ""
    daydir.mkdir(parents=True, exist_ok=True)
    data_file = daydir / f"data_{index}.pickle.gz"
    stat_file = daydir / f"stats_{index}.json.gz"

    logging.debug("Initialize memory.")
    mem = memcls(index=index, workers=config.workers)

    last_data_file = last_stats.get('data_file')
    if last_data_file is not None:
        logging.debug(
            "Loading previous data from `%s`.", last_data_file)
        mem.data = eh.data.read_pickle(last_data_file)
    else:
        logging.debug("No previous data available.")

    # Hotfix: load model after data to always update predictions.
    last_model_dir = last_stats.get('model_dir')
    if last_model_dir is not None:
        logging.debug(
            "Loading previous model from `%s`.", last_model_dir)
        mem.update_predictor(last_model_dir)
    else:
        logging.debug("No previous model available.")

    try:
        inout = load_inout(data_day, index, config=config)
    except FileNotFoundError as error:
        raise NoDataError(f"No data for day `{data_day}`.") from error

    logging.debug("Update memory.")
    # Extra info for analysis: add label to see how old data is.
    label = pd.to_datetime(np.min(inout['ts'])).date().strftime(config.datefmt)
    mem.insert(
        x=inout['in'],
        y=inout['out'],
        label=np.repeat(label, len(inout['in'])),
    )
    current_data = mem.get()
    logging.debug("Saving selected data to `%s`.", data_file)
    eh.data.to_pickle(mem.data, data_file)

    logging.debug("Checking coverage increase.")
    last_train_data_file = last_stats.get("train_data_file")
    if last_train_data_file is not None:
        logging.debug(
            "Loading last training data from `%s`.", last_train_data_file)
        train_data = merge_datadicts(eh.data.read_pickle(last_train_data_file))
        coverage_increase, coverage_stats = _get_memento(mem).coverage_change(
            train_data, current_data=current_data,
            return_stats=True, return_densities=True,
        )
        logging.debug("Coverage increase: `%.2f`%%",
                      coverage_increase * 100)
    else:
        logging.debug("No previous data, 100%% increase!")
        coverage_increase, coverage_stats = 1.0, {}

    retrain = coverage_increase >= threshold
    if retrain:
        logging.info("Training (increase: %.3f).", coverage_increase)
        # New paths for train data and model (in todays dir).
        train_data_file = data_file
        model_dir = daydir / config.model_pathname
        models.train(current_data['x'], current_data['y'],
                     modeldir=model_dir, refmodeldir=config.fugu_feb_dir,
                     index=index, config=config)
        mem.update_predictor(model_dir)
    else:
        logging.info("No training (increase: %.3f).", coverage_increase)
        # Point train data and model to same paths as last day.
        train_data_file = last_train_data_file
        model_dir = last_model_dir

    logging.debug("Compiling statistics.")
    stats = dict(
        day=config.daystr(day),
        data_file=str(data_file),
        model_dir=str(model_dir),
        train_data_file=str(train_data_file),
        last_train_data_file=str(last_train_data_file),
        retrain=bool(retrain),
        **{key: str(val) for key, val in coverage_stats.items()},
    )
    eh.data.to_json(stats, stat_file)
    return stats


def _get_memento(mem: memory.PufferMemory) -> memory.PufferMemento:
    """Memory may be Memento or a Multimemory containing Memento."""
    for _mem in mem.memories:
        if isinstance(_mem, memory.PufferMemento):
            return _mem
    raise RuntimeError("No Memento found!")


def _sync_git(*, model_dir: PathLike, git_dir: PathLike, index: int,
              logger, reraise: bool = False):
    """Copy model files; then commit and push git.

    If indices is None, copy all files.
    Unless `reraise`, only issue a warning if pushing fails.
    """
    model_dir, git_dir = Path(model_dir), Path(git_dir)
    assert model_dir.exists()
    assert git_dir.exists()
    logger.debug("Copy models `%i` from `%s` to `%s` and sync.",
                 index, model_dir, git_dir)

    # Helper commands.
    def _git(*cmds):
        _cmd = ["git", '-C', str(git_dir)] + [str(cmd) for cmd in cmds]
        logger.debug("Git:  `%s`", " ".join(_cmd))
        subprocess.run(_cmd, capture_output=True, check=True)

    def _copy_and_add(filename):
        logger.debug("Copy: `%s`.", filename)
        shutil.copyfile(model_dir / filename, git_dir / filename)
        _git("add", filename)

    # Go :)
    with lockfile(git_dir / "memento.lock"):
        try:
            _git("pull")
            _copy_and_add(f"py-{index}.pt")
            _copy_and_add(f"cpp-{index}.pt")
            _copy_and_add(f"cpp-meta-{index}.json")
        except subprocess.CalledProcessError as error:
            if error.stdout:
                logger.critical(error.stdout.decode("utf-8"))
            if error.stderr:
                logger.critical(error.stderr.decode("utf-8"))
            raise error
        try:
            _git("commit", "-m", f"Update weights for model {index}.")
        except subprocess.CalledProcessError as error:
            logger.debug("No changes.")
            logger.debug(error.stderr.decode("utf-8"))
            if reraise:
                raise error
        else:
            try:
                _git("push")
            except subprocess.CalledProcessError as error:
                logger.warning("Could not push to git.")
                logger.debug(error.stderr.decode("utf-8"))
                if reraise:
                    raise error


@contextlib.contextmanager
def lockfile(filename):
    """File-based lock."""
    with open(filename, "w") as file:  # pylint: disable=unspecified-encoding
        fcntl.flock(file, fcntl.LOCK_EX)
        yield
        fcntl.flock(file, fcntl.LOCK_UN)
