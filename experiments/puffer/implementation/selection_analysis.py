"""Experiment to analyse the selected memory batches."""

import logging

import experiment_helpers as eh
import numpy as np
import pandas as pd
from experiment_helpers.data import Path

from ..config import PufferExperimentConfig
from . import models
from .data import load_inout


def analyse_selection(*, start, end, n_samples, memorycls,
                      config=None,
                      batch_memorycls=None, model_index=0):
    """Highlight the selection impact.

    - Start with the FuguFeb (one-shot) model and take a given span of data.
    - Use a given memory (Memento, Loss-based, ...) to select data.
    - Retrain the model on the selected data.
    - Evaluate the performance of the one-shot and retrained model on the
      selected data.
    """
    config = PufferExperimentConfig.with_updates(config)
    _w = config.workers

    # Main memory.
    try:
        mem = memorycls(workers=_w)
    except TypeError:
        # Not all memories use multiprocessing.
        mem = memorycls()
    mem.update_predictor(config.fugu_feb_dir)

    # Batch memory.
    if batch_memorycls is None:
        batch_mem = mem
    else:
        batch_mem = batch_memorycls()
        batch_mem.update_predictor(config.fugu_feb_dir)

    logging.info("Collect data.")
    _valid_days = _get_valid_days(start, end, model_index, config)
    _x, _y = None, None
    for i, day in enumerate(_valid_days, 1):
        logging.debug("Loading %i/%i", i, len(_valid_days))
        inout = load_inout(day, model_index, config=config)
        if _x is None:
            _x = inout['in']
            _y = inout['out']
        else:
            _x = np.concatenate([_x, inout['in']])
            _y = np.concatenate([_y, inout['out']])

        if len(_x) >= n_samples:
            _x = _x[:n_samples]
            _y = _y[:n_samples]
            break
    else:
        raise ValueError("Not enough data to select from.")

    logging.info("Select data.")
    mem.insert(x=_x, y=_y)

    logging.info("Training.")
    modeldir = Path("./predictor")
    selection = mem.get()

    models.train(selection['x'], selection['y'], modeldir,
                 refmodeldir=config.fugu_feb_dir, index=model_index, config=config)

    logging.info("Evaluation.")
    model = models.load_model(modeldir / f"py-{model_index}.pt")
    refmodel = models.load_model(config.fugu_feb_dir / f"py-{model_index}.pt")
    predictions_ref = models.predict_probabilities(refmodel, _x, workers=_w)

    score = models.predict_logscores(model, _x, _y, workers=_w)
    score_ref = models.predict_logscores(refmodel, _x, _y, workers=_w)

    loss = models.predict_losses(model, _x, _y, workers=_w)
    loss_ref = models.predict_losses(refmodel, _x, _y, workers=_w)

    logging.info("Compute metrics.")
    eval_data = {
        # Info for semantic batching.
        'x': _x,
        'y': _y,
        'ydiscrete': models.ttp_discretize(_y, model),
        'yhat': predictions_ref,  # To get density _before_ selection.
        # Extra metrics.
        'loss': loss,
        'loss_ref': loss_ref,
        'logscore': score,
        'logscore_ref': score_ref,
        # Also record which samples were selected.
        'selected': selected({'x': _x, 'y': _y}, selection),
    }
    if hasattr(batch_mem, 'memories'):
        # Hacky way to deal with MultiMemory.
        # Works for our setup where Memento is always the first Memory.
        batch_mem = batch_mem.memories[0]

    batches = batch_mem.batch(eval_data)

    output_metrics = [
        'loss', 'loss_ref', 'logscore', 'logscore_ref', 'selected']
    results = {
        metric: np.array([np.mean(batch[metric]) for batch in batches])
        for metric in output_metrics
    }

    matrices = batch_mem.compute_matrices(batches)
    results['density'] = batch_mem.agg([mat.mean(axis=1) for mat in matrices])

    logging.info("Store results.")
    eh.data.to_npz(results, "results.npz")  # Stores a dict of arrays.


def _get_xy_bytes(datadict):
    """Turn x and y per sample into a single byte string for easy comparison"""
    return np.array([_x.tobytes() + _y.tobytes()
                     for _x, _y in zip(datadict['x'], datadict['y'])])


def selected(datadict, selection):
    """For each point in datadict, check whether its in selection."""
    bytes_all = _get_xy_bytes(datadict)
    bytes_selection = _get_xy_bytes(selection)

    return np.isin(bytes_all, bytes_selection)


def _get_valid_days(start, end, index, config):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    logging.debug("Checking data between `%s` and `%s`,",
                  config.daystr(start), config.daystr(end))
    valid_days = []
    invalid = []
    all_days = pd.date_range(start, end, freq='D')
    for day in all_days:
        daystr = config.daystr(day)
        daydir = config.get_data_directory(day)
        inout_file = daydir / config.inout_filenames[index]
        if not inout_file.is_file():
            invalid.append(f"{daystr}: missing inout.")
        else:  # All data is ok, and we should return path to inout.
            valid_days.append(day)

    if invalid:
        logging.warning("Missing data for `%i` day(s).", len(invalid))
        logging.debug("\n".join(invalid))

    return valid_days
