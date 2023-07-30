#!/usr/bin/env python
"""Run a demo subset of the real experiments.

This demo illustrates how the experiments are organized.
See `experiments/puffer/experiments.py` for the full configuration, and note
that the full configuration is using the same functions, just with different
date ranges.
"""
# pylint: disable=invalid-name

from functools import partial

import click
import experiment_helpers as eh
import pandas as pd

from experiments.puffer import config
from experiments.puffer.implementation import data, memory, replay, analysis

default_memory = partial(
    memory.PufferMemory,
    size=1000000, batching_size=256,
    bw=0.1, temperature=0.01,
    insert_chunksize=2000000,  # Optional, limits maximum memory usage.
)

# First Monday--Sunday in July 2023.
demo_start_day = "2023-07-03"
demo_end_day = "2023-07-09"
day_range = pd.date_range(demo_start_day, demo_end_day, freq="D")

# One group contains "experiments" for downloads.
downloads = {
    day.strftime("%Y-%m-%d"): partial(data.download_data, day=day)
    for day in day_range
}

# Another group contains the actual experiments: analysis and a replay demo.
experiments = {
    f"analysis/{day.strftime('%Y-%m-%d')}":
    partial(analysis.analyze_day, day=day) for day in day_range
}
experiments['replay'] = partial(
    replay.evaluate,
    startday=demo_start_day, endday=demo_end_day,
    replaycls=replay.PufferDataReplay, memorycls=default_memory,
)


# CLI.
# ====

@click.group()
def demo_cli():
    """Demo CLI for experiments.

    The main functions are automatically added to the default CLI, so we
    create a separate one here for demo purposes only.
    """


download_group = eh.framework.ParametrizedExperiments(
    "download", downloads, cli=demo_cli,
    cli_help="Download Puffer data from 2022-07-20 to 2022-07-27.",
    configcls=config.PufferExperimentConfigNoGPU)
experiments_group = eh.framework.ParametrizedExperiments(
    "run-demo", experiments, cli=demo_cli,
    cli_help="Evaluate downloaded data and run a replay experiment.",
    configcls=config.PufferExperimentConfig)


if __name__ == "__main__":
    demo_cli()  # pylint: disable=no-value-for-parameter
