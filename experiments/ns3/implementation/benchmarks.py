"""Microbenchmarks using the simulation data."""

import gc
import logging
import time
from itertools import chain

import experiment_helpers as eh
import numpy as np
import pandas as pd

from memento import alternatives

from ..config import PREDICTORTYPE, Ns3ExperimentConfig
from .load_data import load_transtime_data, load_workload_data
from .memory import TranstimeMemory, WorkloadMemory
from .ns3_replay_helper import TranstimeDataReplay, WorkloadDataReplay


def factory(predictortype: PREDICTORTYPE):
    """Return appropriate loaders, framework, and memory class."""
    if predictortype == "workload":
        return (
            load_workload_data,
            WorkloadDataReplay,
            WorkloadMemory
        )
    if predictortype == "transtime":
        return (
            load_transtime_data,
            TranstimeDataReplay,
            TranstimeMemory
        )
    else:
        raise ValueError("Unknown predictor type.")


# Note: `*` syntax means that all arguments are keyword arugments!
def performance_benchmark(*, config=None,
                          predictortype: PREDICTORTYPE,
                          scenarios, iterations: int,
                          **mem_params
                          ):
    """Run performance benchmarks."""
    config = Ns3ExperimentConfig.with_updates(config)

    predictor = baseline_predictor(predictortype, scenarios, config)

    results, stats = [], []
    for iteration in range(iterations):
        logging.info("Iteration: `%i` of `%i`", iteration+1, iterations)

        _, replaycls, memcls = factory(predictortype)

        # Use a baseline predictor so we can test all features.
        mem = memcls(config=config, **mem_params)
        mem.update_predictor(predictor)

        replay = replaycls(
            scenarios,
            config=config,
            train_runs=config.benchmark_runs,
            eval_runs=config.eval_runs,
            memory=mem,
            out=None, logger=None,  # Only one iteration, don't keep track.
        )
        _results, _stats = replay.run()

        # Add metadata and add to other results.
        results.append(_results.assign(iteration=iteration, **mem_params))
        stats.append(_stats.assign(iteration=iteration, **mem_params))

    logging.info("Storing results.")
    combined_results = pd.concat(results, ignore_index=True)
    combined_stats = pd.concat(stats, ignore_index=True)
    eh.data.to_pickle(combined_results, 'results.pickle.gz')
    eh.data.to_pickle(combined_stats, 'stats.pickle.gz')


# Note: `*` syntax means that all arguments are keyword arugments!
def time_benchmark(*,
                   predictortype: PREDICTORTYPE,
                   memsize: int,
                   samplesize: int,
                   workers: int,
                   iterations: int,
                   batchsize: int,
                   scenario,
                   config=None):
    """Benchmark memory computation time."""
    config = Ns3ExperimentConfig.with_updates(config)
    config.workers = workers
    assert memsize < np.min(samplesize)

    rng = np.random.default_rng(config.random_state)

    dataloader, _, memcls = factory(predictortype)

    logging.info("Loading data.")

    x_runs, y_runs, _ = zip(*chain.from_iterable(
        dataloader(exp, runs=config.runs, config=config)
        for exp, _ in scenario[0]))
    full_data = {'x': np.concatenate(x_runs), 'y': np.concatenate(y_runs)}

    logging.info("Benchmarking time.")
    logging.info("(%i worker(s)) %s: Samplesize: %i, Batchsize: %i.",
                 workers, predictortype, samplesize, batchsize)
    results = []

    # Pre-predict, we don't want to measure model-dependent prediction time.
    # This is a bit hacky, but we use the memory to predict because there
    # we have already configured how to do it.

    predictor = baseline_predictor(predictortype, scenario, config)
    memory = memcls()
    memory.update_predictor(predictor)
    # The memory is a multimemory, Memento is the first sub-memory.
    full_data = memory.memories[0].add_predictions(full_data)

    for iteration in range(iterations):
        mem = memcls(
            size=memsize,
            batching_size=batchsize,
            config=config,
        )
        sample_index = rng.choice(len(full_data['x']),
                                  samplesize, replace=False)
        data_sample = {k: v[sample_index] for k, v in full_data.items()}

        # Time the coverage algorithm.
        start = time.time()
        mem.insert(data_sample)
        time_coverage = time.time() - start

        results.append({
            "type": predictortype,
            "workers": workers,
            "iteration": iteration,
            "memsize": memsize,
            "samplesize": samplesize,
            "batchsize": batchsize,
            "time": time_coverage,
        })
        gc.collect()  # Limit how hoch memory accumulates.
    combined = pd.DataFrame(results)
    eh.data.to_pickle(combined, 'results.pickle.gz')


def baseline_predictor(predictortype: PREDICTORTYPE, scenario,
                       config: Ns3ExperimentConfig):
    """Train a baseline predictor.

    Nothing fancy, just something to get predictions to properly test timing.
    """
    _, replaycls, _ = factory(predictortype)
    mem = alternatives.ReservoirMemory()  # only 1 iteration, no forgetting.

    # Get unique experiments without size limits (bit hacky implementation).
    experiments = replaycls(scenario, memory="placeholder").unique_exps()

    # With unique experiments, replay one iteration and fetch model afterwards.
    replay = replaycls(
        [experiments],
        memory=mem,
        train_runs=config.benchmark_baseline_runs,
        eval_runs=(1,),  # We don't need eval, so this doesn't matter.
        balance=True,
        config=config,
        out=None, logger=None,  # Only one iteration, don't keep track.
    )
    replay.run()
    return replay.predictor
