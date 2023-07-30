"""Unrelated to everything else, just provide some numbers on the simulation"""

import logging

import pandas as pd
from experiment_helpers.data import Path

import memento

from ..config import Ns3ExperimentConfig
from .load_data import load_transtime_data, load_workload_data


def overview(config):
    """Load all available ns3 data and compute a summary."""
    config = Ns3ExperimentConfig.with_updates(config)
    records = []

    datadir = Path(config.ns3_data_directory)

    logging.info("Loading all experiments")
    experiments = set(_get_name(file) for file in datadir.iterdir())

    n_exps = len(experiments)
    for i, experiment in enumerate(experiments, 1):
        logging.info("%s (%i/%i).", experiment, i, n_exps)
        all_wl = load_workload_data(experiment, config=config)
        all_tt = load_transtime_data(experiment, config=config)

        for run, (workload, transtime) in enumerate(zip(all_wl, all_tt)):
            records.append({
                'experiment': experiment,
                'run': run,
                'horizon': config.horizon,
                'wl_samples': memento.utils.sample_count(workload),
                'tt_samples': memento.utils.sample_count(transtime),
            })

    frame = pd.DataFrame.from_records(records).set_index(["experiment", 'run'])
    frame.to_csv('overview.csv.gz')


def _get_name(filename):
    name_and_runs = filename.stem
    return '_'.join(name_and_runs.split('_')[:-1])
