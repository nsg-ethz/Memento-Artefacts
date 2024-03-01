"""Data replay utility subclassed for ns3.

That is, the data loading, train, and prediction methods are implemented
to work with the ns-3 data and models.

Running everything, including storing models, data, results, ..., is done
by the DataReplay class already.
"""
# pylint: disable=unused-argument, import-outside-toplevel
# Unavoidable when implementing the methods. Also we avoid importing model
# because importing tensorflow is really slow.


import abc
from collections import defaultdict

import numpy as np
import pandas as pd
from experiment_helpers.typing import (ArrayLike, Dict, Iterable, List,
                                       Sequence, Tuple, Union)

import memento
from experiments.ns3.implementation import load_data

from ... import replay_helper
from ..config import Ns3ExperimentConfig
from . import load_data

# List of iterations -> List of ns3 experiments, either specified by name
# or as a tuple (name, max_samples).
ScenarioType = Sequence[Sequence[Union[str, Tuple[str, int]]]]


class Ns3DataReplay(replay_helper.ReplayFramework):
    """Data replay utility for ns3 experiments."""

    def __init__(self, scenario: ScenarioType, *args,
                 multi_mem_filter=None,
                 config=None, balance: bool = False,
                 train_runs: Sequence[int] = None,
                 eval_runs: Sequence[int] = None,
                 **kwargs):
        self.config = Ns3ExperimentConfig.with_updates(config)
        self.scenario = scenario
        self.balance = balance
        self.train_runs = train_runs
        self.eval_runs = eval_runs
        self.rng = np.random.default_rng(self.config.random_state)

        if multi_mem_filter is None:
            multi_mem_filter = [0]  # Coverage memory is first one usually.

        super().__init__(*args, multi_mem_filter=multi_mem_filter, **kwargs)

    def unique_exps(self, scenario=None, **kwargs):
        """Return only the unique experiments across runs."""
        if scenario is None:
            scenario = self.scenario
        _flat = []
        for run in scenario:
            for exp in run:
                if isinstance(exp, tuple):
                    _flat.append(exp[0])
                else:
                    _flat.append(exp)
        return np.unique(_flat, **kwargs)

    def load_data(self, starting_iteration=0):
        """Use one run as test data, load other as train data.

        We split the test data by each unique experiment in the scenario
        for easier side-by-side analysis.
        """
        all_iter_data = self.load_by_experiment(
            self.scenario, self.train_runs, balance=self.balance)

        # Balanced evaluation; same number of runs per unqie ns3 experiment.
        eval_experiments = np.repeat(
            self.unique_exps(), len(self.eval_runs))
        eval_data = self.load_by_experiment(
            [eval_experiments], self.eval_runs,
        )[0]  # Only one iteration in the eval scenario.

        # Track which experiments have appeared so far.
        observed = {experiment: False for experiment in eval_data}

        for iteration, iter_data in enumerate(all_iter_data):
            # Also track experiments appeared this iteration.
            observednow = {experiment: False for experiment in eval_data}
            # Update which experiments we have observed so far.
            for experiment in iter_data:
                observed[experiment] = True
                observednow[experiment] = True

            if iteration < starting_iteration:
                continue

            # Merge and shuffle the datadicts together.
            input_data = self._shuffle_sample(
                memento.utils.merge_datadicts(iter_data.values()))
            assert memento.utils.sample_count(input_data) > 0
            assert memento.utils.sample_count(eval_data) > 0
            yield (input_data, (eval_data, observed.copy(), observednow))

    def load_by_experiment(self, scenario, runs, balance=False) \
            -> List[Dict[str, Dict[str, ArrayLike]]]:
        """For each iteration in scenario, return data per experiment.

        If `balance`, ensure the number of samples of each experiment is equal
        over all iterations.
        """
        # Step 1: Check how many runs of which ns3 experiment we need,
        # And load as many into a dict mapping ns3 experiment to a list of
        # data dicts.
        _data: Dict[str, Dict[str, Sequence[Dict[str, ArrayLike]]]] = {}
        for experiment, counts in \
                zip(*self.unique_exps(scenario=scenario, return_counts=True)):
            # Don't need to load more than `counts`.
            run_data = self.load_runs(experiment, runs[:counts])
            _data[experiment] = iter(run_data)

        # Step 2, go through the scenario and fetch the ns3 experiments
        # needed at each iteration. Multiple runs from the same experiment
        # may be needed at the same iteration.

        # Now combine training runs as requested.
        all_iter_data = []
        for exps in scenario:
            # Collect the ns3 experiment runs needed at this iteration.
            # We may need to balance them later, so store them separately.
            exp_data = defaultdict(list)
            for experiment in exps:
                if isinstance(experiment, tuple):
                    name, max_samples = experiment
                else:
                    name, max_samples = experiment, None
                run_data = next(_data[name])
                sample = self._shuffle_sample(run_data, max_samples)
                exp_data[name].append(sample)

            # Merge the individual arrays together to one big array.
            # Still keep them separated by ns3 experiment for each iteration.
            all_iter_data.append({
                exp: memento.utils.merge_datadicts(datadicts)
                for exp, datadicts in exp_data.items()
            })

        # If balance, ensure the number of samples of each experiment is equal
        # over all iterations.
        if balance:
            # Find the smallest number of samples across all experiments.
            min_samples = min(
                min(memento.utils.sample_count(datadict)
                    for datadict in iter_data.values())
                for iter_data in all_iter_data
            )
            all_iter_data = [
                {exp: self._shuffle_sample(datadict, min_samples)
                 for exp, datadict in iter_data.items()}
                for iter_data in all_iter_data
            ]

        # Return the list of dicts {exp -> datadict}.
        return all_iter_data

    def _shuffle_sample(self, datadict: Dict[str, ArrayLike], n_samples=None):
        """Shuffle arrays, optionally sample (without replacement).

        If n_samples is None, just shuffle in place.
        """
        datadict = memento.utils.check_datadict(datadict)
        data_len = memento.utils.sample_count(datadict)
        if n_samples is None:
            n_samples = data_len
        index = self.rng.choice(data_len, n_samples, replace=False)
        return {key: val[index] for key, val in datadict.items()}

    @abc.abstractmethod
    def load_runs(self, experiment, runs) -> Sequence[ArrayLike]:
        """Load experiment runs."""

    def total_iterations(self):
        return len(self.scenario)

    def save_predictor(self, predictor, path):
        """Saving is independent of model type."""
        from . import models
        models.save_model(predictor, path)

    @abc.abstractmethod
    def load_predictor(self, path):
        """Pickle load does not work properly; overwrite per model."""

    def evaluate(self, predictor, eval_data) -> pd.DataFrame:
        data_per_exp, observed, observednow = eval_data
        return self.compute_metrics(
            predictor, data_per_exp, observed, observednow)

    @abc.abstractmethod
    def compute_metrics(self, predictor, data_per_exp, observed, observednow) \
            -> Dict[str, float]:
        """Compute metrics for a single experiment."""


class WorkloadDataReplay(Ns3DataReplay):
    """Retraining framework for the workload model."""

    def load_runs(self, experiment, runs):
        return load_data.load_workload_data(
            experiment, config=self.config, runs=runs)

    def compute_metrics(self, predictor, data_per_exp, observed, observednow):
        """Accuracy and logscore, separate per experiment.

        (Easier to interpret whats happending)
        """
        # pylint: disable=import-outside-toplevel
        from .models import predict_workload
        metrics = []
        for exp, datadict in data_per_exp.items():
            eval_x, eval_y = datadict['x'], datadict['y']
            probabilities = predict_workload(predictor, eval_x)
            # For each row (arange), get the probability of the workload.
            # where workload 0 is column 0, etc.
            with np.errstate(invalid='ignore'):
                logscore = np.log(
                    probabilities[np.arange(len(eval_y)), eval_y])
            accuracy = np.mean(probabilities.argmax(axis=1) == eval_y)

            metrics.append({
                "experiment": exp,
                "observed": observed[exp],
                "observed_now": observednow[exp],
                "accuracy": accuracy,
                "logscore-mean": np.nanmean(logscore),
                "logscore-median": np.nanquantile(logscore, 0.5),
                "logscore-1st": np.nanquantile(logscore, 0.01)
            })

        return pd.DataFrame(metrics).astype({"experiment": "category"})

    def compute_stats(self, iteration, memory_update_time, mem_data, eval_data):
        """Add workload counts to stats."""
        do_train, stats = super().compute_stats(
            iteration, memory_update_time, mem_data, eval_data)

        # Use y (ground truth) to count workloads in selection
        workloads = mem_data['y']
        for workload in range(self.config.workloads):
            stats[f"w{workload + 1}"] = np.sum(workloads == workload)

        return do_train, stats

    def train(self, traindata, last_predictor=None):
        """Training with different reference models."""
        from . import models  # pylint: disable=import-outside-toplevel
        return models.train_workload_model(
            traindata['x'], traindata['y'], config=self.config)

    def load_predictor(self, path):
        """Load workload model."""
        from . import models  # pylint: disable=import-outside-toplevel
        return models.load_workload_model(path, config=self.config)


class TranstimeDataReplay(Ns3DataReplay):
    """Retraining framework for the transtime model."""

    def load_runs(self, experiment, runs):
        return load_data.load_transtime_data(
            experiment, config=self.config, runs=runs)

    def compute_metrics(self, predictor, data_per_exp, observed, observednow):
        """Absolute and Squared errors.

        Various percentiles for all data currently observed.
        (i.e. during congestion how well does it work?)
        """
        all_errors = []
        metrics = []

        for exp, datadict in data_per_exp.items():
            eval_x, eval_y = datadict['x'], datadict['y']
            if not observednow[exp]:  # if not observed[exp]:
                continue
            predictions = predictor.predict(eval_x).flatten()
            errors = predictions - eval_y
            all_errors.append(errors)
            metrics.extend(self._error_metrics(exp, errors))

        combined = np.concatenate(all_errors)
        metrics.extend(self._error_metrics("all", combined))

        min_len = min(len(arr) for arr in all_errors)
        sampled = np.concatenate([
            self.rng.choice(arr, size=min_len, replace=False)
            for arr in all_errors
        ])
        metrics.extend(self._error_metrics("balanced", sampled))

        return pd.DataFrame.from_records(metrics)

    def _error_metrics(self, experiment, errors):
        abs_errors = np.abs(errors)
        metrics = [
            {
                'experiment': experiment,
                "metric": "count",
                "value": len(abs_errors)
            },
            {
                'experiment': experiment,
                "metric": "MAE",
                "value": np.mean(abs_errors)
            },
            {
                'experiment': experiment,
                "metric": "MSE",
                "value": np.mean(np.array(abs_errors)**2)
            }
        ]
        for quantile in [0.5, 0.9, 0.99, 0.999]:
            metrics.append({
                'experiment': experiment,
                "metric": quantile,
                "value": np.quantile(abs_errors, quantile)
            })
        return metrics

    def train(self, traindata, last_predictor=None):
        """Training with different reference models."""
        from . import models  # pylint: disable=import-outside-toplevel
        return models.train_transtime_model(
            traindata['x'], traindata['y'], config=self.config)

    def load_predictor(self, path):
        """Load transtime model."""
        from . import models  # pylint: disable=import-outside-toplevel
        return models.load_transtime_model(path, config=self.config)
