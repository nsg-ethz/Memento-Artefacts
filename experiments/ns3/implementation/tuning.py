"""Hyperparameter tuning."""
# pylint: disable=invalid-name

import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import sklearn
import skopt

from ..config import Ns3ExperimentConfig
from . import load_data


def ns3_tune(*, predictortype, config=None):
    """Find the optimal model params."""
    config = Ns3ExperimentConfig.with_updates(config)

    # Assign train and eval functions and load data.
    # pylint: disable=import-outside-toplevel
    from . import models
    logging.info("Tuning parameters for `%s` model", predictortype)
    if predictortype == 'workload':
        train_func = models.train_workload_model
        eval_func = negative_balanced_accuracy
        load_func = load_data.load_workload_data
        experiments = ["w1", "w2", "w3"]

    elif predictortype == 'transtime':
        train_func = models.train_transtime_model
        eval_func = mse
        load_func = load_data.load_transtime_data
        experiments = [
            "w1", "w2", "w3",
            "w1_c100", "w2_c100", "w3_c100",
            "w1_c133", "w2_c133", "w3_c133",
        ]

    else:
        raise RuntimeError(f"Unknown predictor type: `{predictortype}`.")

    logging.info("Loading data.")
    data_x, data_y = balance_and_merge([
        load_func(
            experiment, runs=config.tuning_runs, config=config)
        for experiment in experiments
    ], random_state=config.random_state)

    logging.info("Tuning.")
    result = _tune(data_x, data_y, train_func, eval_func, config)
    skopt.dump(result, f'{predictortype}_params.pickle', store_objective=False)


def _tune(data_x, data_y, train_func, eval_func, config: Ns3ExperimentConfig):
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    func = partial(_train_and_eval,
                   train_func=train_func, eval_func=eval_func)

    # The test space.
    space = [
        skopt.space.Integer(3, 10, name="hidden_layers"),
        skopt.space.Integer(128, 512, name="hidden_units"),
        skopt.space.Real(10**-7, 10**0, "log-uniform", name='learning_rate'),
    ]

    splitter = sklearn.model_selection.ShuffleSplit(
        n_splits=config.tuning_splits, train_size=config.tuning_train_size,
        random_state=config.random_state)

    # If we have multiple GPUs available, we can parallelize.
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    max_parallel = min(num_gpus, config.workers)

    # The function to minimize.
    @skopt.utils.use_named_args(space)
    def eval_params(**kwargs):
        jobs = []
        for train_index, test_index in splitter.split(data_x):
            train_x, train_y = data_x[train_index], data_y[train_index]
            test_x, test_y = data_x[test_index], data_y[test_index]
            jobs.append((train_x, train_y, test_x, test_y, kwargs))

        if max_parallel > 1:
            # Parallelize for more GPU jobs.
            # Ensure we spawn instead of fork to avoid deadlock; see:
            # https://github.com/keras-team/keras/issues/9964
            with mp.get_context("spawn").Pool(max_parallel) as pool:
                results = pool.starmap(func, jobs)
        else:
            results = [func(*job) for job in jobs]

        result = np.nanmean(results)
        return result if np.isfinite(result) else config.tuning_penalty

    # Optimize
    return skopt.gp_minimize(eval_params, space,
                             n_calls=config.tuning_iterations,
                             random_state=config.random_state,
                             verbose=(config.loglevel <= logging.INFO))


def _train_and_eval(train_x, train_y, test_x, test_y, train_kwds,
                    train_func=None, eval_func=None):
    try:
        model = train_func(train_x, train_y, **train_kwds)
        return eval_func(model, test_x, test_y)
    except TypeError:
        # EarlyStopping has a bug and crashed if hyperparams cause
        # performance to only degrade. Skip this (rare) case.
        # Related to:
        # https://github.com/tensorflow/tensorflow/issues/45187
        return np.NaN


def balance_and_merge(experiments_and_runs, random_state=42):
    """From a nested list (experiments, runs), extract and merge x and y."""
    rng = np.random.default_rng(random_state)
    lengths = []
    for runs in experiments_and_runs:
        for x, _, _ in runs:
            lengths.append(len(x))
    min_len = min(lengths)

    all_x, all_y = [], []
    for runs in experiments_and_runs:
        for x, y, _ in runs:
            index = rng.choice(len(x), min_len, replace=False)
            all_x.append(x[index])
            all_y.append(y[index])
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


# Scoring functions.
# ==================

def negative_balanced_accuracy(model, test_x, test_y):
    """Returned balances accuracy (negative to make it a minimization)."""
    # For each x, check if correct workload is predicted.
    right_label = model.predict(test_x).argmax(axis=1) == test_y

    accuracy_per_label = []
    for label in np.unique(test_y):
        # First average only within class.
        accuracy_per_label.append(np.mean(right_label[test_y == label]))

    return - np.mean(accuracy_per_label)


def mse(model, test_x, test_y):
    """Return the mean square error (already minimzation)."""
    predicted_transtime = model.predict(test_x).flatten()  # flatten for 1-d.
    return np.mean((predicted_transtime - test_y)**2)
