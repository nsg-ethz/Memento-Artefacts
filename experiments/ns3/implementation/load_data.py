"""Load ns3 simulation results and format into input, output, and appids."""

import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from experiment_helpers.data import Path

from ..config import Ns3ExperimentConfig

logger = logging.getLogger("experiments.ns3.load_data")


def load_workload_data(
        experiment: str, config=None, runs: Sequence[int] = None
) -> List[Dict[str, np.ndarray]]:
    """Load and prepare workload data across experiment runs."""
    config = Ns3ExperimentConfig.with_updates(config)
    datadir = Path(config.ns3_data_directory)
    runs = config.runs if runs is None else runs
    horizon = config.horizon

    logger.debug("Loading `workload` data for "
                 "`%s` (runs: `%s`, horizon: `%i`) from `%s`",
                 experiment, runs, horizon, datadir)
    return [
        load_run_workload(datadir / f"{experiment}_{run}.csv", horizon)
        for run in runs
    ]


def load_transtime_data(
        experiment: str, config=None, runs: Sequence[int] = None
) -> List[Dict[str, np.ndarray]]:
    """Load and prepare transtime data across experiment runs."""
    config = Ns3ExperimentConfig.with_updates(config)
    datadir = Path(config.ns3_data_directory)
    runs = config.runs if runs is None else runs
    horizon = config.horizon

    logger.debug("Loading `transtime` data for "
                 "`%s` (runs: `%s`, horizon: `%i`) from `%s`",
                 experiment, runs, horizon, datadir)
    return [
        load_run_transtime(datadir / f"{experiment}_{run}.csv", horizon)
        for run in runs
    ]


def load_run_workload(path, horizon):
    """Load a run and prepare for workload classification.

    Return x, y, appids, where:
    x: current size and past `horizon-1` sizes.
    y: workload.
    """
    frame = load_frame(path)[['size', 'workload', 'application']]
    shifted = multi_shifts(frame, shifts=horizon-1, groupby="application")

    data_cols = data_cols = [f"size_{i}" for i in range(horizon)]
    sizes = shifted[data_cols].to_numpy()
    apps = shifted['application_0'].to_numpy()
    workloads = shifted['workload_0'].to_numpy()

    return {
        "x": sizes,
        "y": workloads,
        "appids": apps,
    }


def load_run_transtime(path, horizon):
    """Load a run and prepare for transmit time prediction.

    Return x, y, appids, where:
    x: current size and both past `horizon-1` sizes and transmit times.
    y: current transmit time.
    """
    frame = load_frame(path)[['size', 'trans_time', 'workload', 'application']]
    shifted = multi_shifts(frame, shifts=horizon-1, groupby="application")

    # Do not include current trans time -- this is the target.
    data_cols = ([f"size_{i}" for i in range(horizon)] +
                 [f"trans_time_{i}" for i in range(1, horizon)])
    sizes_and_times = shifted[data_cols].to_numpy()
    apps = shifted['application_0'].to_numpy()
    current_time = shifted['trans_time_0'].to_numpy()

    return {
        "x": sizes_and_times,
        "y": current_time,
        "appids": apps,
    }


def load_frame(path: Path):
    """Load csv into dataframe."""
    if not Path(path).exists():
        logging.debug("`%s` does not exist, trying compressed version.", path)
        path = Path(path).with_suffix(".csv.gz")
    columns = ['t', 'trans_time', 'size', 'workload', 'application']
    frame = (
        pd
        .read_csv(path, names=columns)
        .sort_values(by="t")
        # A bit hacky, but workloads are numbered 1..3, zero-indexed ones.
        .assign(workload=lambda frame: frame["workload"] - 1)
        .astype({
            't': float,
            'trans_time': float,
            'size': int,
            'workload': 'category',
            'application': 'category',
        })
    )
    return frame


def multi_shifts(frame, shifts: int = 0, groupby: str = None,
                 dropna: bool = True) -> pd.DataFrame:
    """Return shifted copies of the dataframe, optionally by group.

    I.e. a frame with columns a, b and shifts=2 will be turned into a frame with
    a_0, b_b, a_1, b_1, a_2, b_2; the original column is suffixed by 0, and the
    new columns by the number of rows that have been shifted.

    Note that this will result in columns with NaN unless `dropna` is True.
    """
    if groupby is not None:
        # .groupby.shift seems to always swallow the groupby column, ignoring
        # "as_index" and similar. so use a helper column that will vanish.
        frame = frame.assign(grouper=frame[groupby]).groupby("grouper")
    shifted = pd.concat(
        [frame.shift(i).add_suffix(f"_{i}") for i in range(shifts+1)],
        axis=1,
    )
    return shifted.dropna() if dropna else shifted
