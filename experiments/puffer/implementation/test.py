"""A tiny experiment that tests loading data, training, and storing results."""

import logging

import experiment_helpers as eh
import pandas as pd
from experiment_helpers.data import Path

from ..config import PufferExperimentConfig
from . import data, models


def test_training(*, day=pd.to_datetime("2022-07-27"), index=0, config=None):
    """Do a short test run of training a puffer model from data."""
    data.download_data(day=day, config=config)  # Ensure we have the data.
    config = PufferExperimentConfig.with_updates(config)

    # Load data
    logging.info("Loading data for day `%s`.", config.daystr(day))
    inoutpath = config.puffer_data_directory / config.daystr(day) / \
        config.inout_filenames[index]
    inout = eh.data.read_npz(inoutpath)

    # Train model
    logging.info("Training model.")
    modeldir = Path("./model")

    n_samples = 100  # Small training set for testing.
    x_train, y_train = inout["in"][:n_samples], inout["out"][:n_samples]
    models.train(
        x_train, y_train, modeldir,
        refmodeldir=config.fugu_feb_dir, index=index,
        config=config, train_args={"train_epochs": 2})

    # Make sure data could we written.
    logging.info("Checking that model was written to `%s`.",
                 modeldir.absolute())
    assert modeldir.is_dir()
    assert list(modeldir.iterdir())
