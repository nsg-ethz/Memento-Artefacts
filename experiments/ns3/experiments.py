"""All ns-3 experiments.

This file defines experiments in the form:

```
experiment_group = "ns3"
experiment_defintions = {
    "<path>": function(config)
}
```

That is, each experiment is identified by the path where its outputs are
stored, and is a function that takes a (user) config as argument.

Concretely, outputs are stored in `config.output_directory / ns3 / <path>`.

Then we pass these definitions to utilities that take care of running the
functions from the appropriate directories, parallelizing, etc.

Below, we use functools.partial to prepare the functions. It allows to
partially set some function arguments (hence the name), so we can easily
parametrize one experiment function in different ways to define a range
of experiments.
"""
# pylint: disable=invalid-name

from functools import partial

import experiment_helpers as eh

from memento import alternatives

from . import config
from .implementation import memory, overview, replay, tuning

# The name is both the command for the CLI, as well as the directory where all
# results will be stored (it will be a subdirectory of config.output_directory).
experiment_group = 'ns3-replay'

# Create a dictionary with all experiments we run.
experiment_definitions = {
    'overview': overview.overview,
    # More are added below!
}


memsize = 20000
memsize_transtime = 60000

# Replay data in different patterns and compare memories.
# =======================================================

# Each scenario is a sequence of workloads; each element in the sequence is the
# data replayed at one iteration. For each iteration, a different (random) run
# of data is used to avoid overlap.

# First, define all memory classes (fix default arguments where needed)

workload_memories = {
    'memento': partial(
        memory.WorkloadMemory, size=memsize,
        distances="both", batching_size=256,
        bw=0.1, temperature=0.01, random_fraction=0.0,
    ),
    'memento-deterministic': partial(
        memory.WorkloadMemory, size=memsize,
        distances="both", batching_size=256,
        bw=0.1, temperature=0, random_fraction=0.0,
    ),
    'fifo': partial(alternatives.FIFOMemory, size=memsize),
    'random': partial(alternatives.ReservoirMemory, size=memsize),
    'lars': partial(memory.WorkloadLARS, size=memsize),
}
transtime_memories = {  # No LARS for regression
    'memento': partial(
        memory.TranstimeMemory, size=memsize_transtime,
        distances="both", batching_size=256,
        bw=0.1, temperature=0.01, random_fraction=0.0,
    ),
    'memento-deterministic': partial(
        memory.TranstimeMemory, size=memsize_transtime,
        distances="both", batching_size=256,
        bw=0.1, temperature=0, random_fraction=0.0,
    ),
    'fifo': partial(alternatives.FIFOMemory, size=memsize_transtime),
    'random': partial(alternatives.ReservoirMemory, size=memsize_transtime),
}

transtime_memories_larger_mem = {  # No LARS for regression
    'memento': partial(
        memory.TranstimeMemory, size=4*memsize,
        distances="both", batching_size=256,
        bw=0.1, temperature=0.01, random_fraction=0.0,
    ),
    'memento-deterministic': partial(
        memory.TranstimeMemory, size=4*memsize,
        distances="both", batching_size=256,
        bw=0.1, temperature=0, random_fraction=0.0,
    ),
    'fifo': partial(alternatives.FIFOMemory, size=4*memsize),
    'random': partial(alternatives.ReservoirMemory, size=4*memsize),
}


# Periodic patterns: w2 all the time, w1 every 5 iterations, w3 every 10.
# -----------------------------------------------------------------------

periodic_scenario = (
    [["w1", "w2", "w3"]] + [["w2"]] * 4 +
    [["w1", "w2"]] + [["w2"]] * 4 +
    [["w1", "w2", "w3"]] + [["w2"]] * 4 +
    [["w1", "w2"]] + [["w2"]] * 4
)
# Use with workload prediction.
for mem, memcls in workload_memories.items():
    experiment_definitions[f'periodic/{mem}'] = partial(
        replay.evaluate, scenario=periodic_scenario,
        predictortype="workload", memorycls=memcls,
    )


# Incremental learning: w1, w2, w3, one after another. Workload again.
# --------------------------------------------------------------------

incremental_scenario = [["w1"]] * 10 + [["w2"]] * 10 + [["w3"]] * 10
for mem, memcls in workload_memories.items():
    experiment_definitions[f'incremental/{mem}'] = partial(
        replay.evaluate, scenario=incremental_scenario,
        predictortype="workload", memorycls=memcls,
    )


# Regression: a scenario with increasing congestion for the transtime model.
# --------------------------------------------------------------------------


low = ["w1", "w2", "w3"]
med = ["w1_c100", "w2_c100", "w3_c100"]
high = ["w1_c133", "w2_c133", "w3_c133"]

congestion_scenario = (
    [low] * 10 +
    [med] * 10 +
    [high] * 10
)
for mem, memcls in transtime_memories.items():
    experiment_definitions[f'congestion/{mem}'] = partial(
        replay.evaluate, scenario=congestion_scenario,
        predictortype="transtime", memorycls=memcls,
    )

for mem, memcls in transtime_memories_larger_mem.items():
    experiment_definitions[f'congestion-larger-mem/{mem}'] = partial(
        replay.evaluate, scenario=congestion_scenario,
        predictortype="transtime", memorycls=memcls,
    )


# Benchmarks.
# ===========
# We are not using these benchmarks anymore, as we have switched to the
# real-life puffer data to benchmark. Uncomment if you want to use it anyways.

# iterations = 10  # Repeat measurements 10 times.

# # Which data to use for the benchmarks.
# benchmark_scenarios = dict(
#     # One run only.
#     workload=[[
#         # Deliberately subsample w3 to make it harder.
#         ("w1", 10000), ("w2", 30000), ("w3", 5000)
#     ]],
#     transtime=[[
#         ("w1", 10000), ("w2", 30000), ("w3", 5000),
#         # And add 50 % of the samples on top for congestion.
#         ("w1_c100", 5000), ("w2_c100", 15000), ("w3_c100", 2500),
#     ]],
# )
#
#
# # Performance benchmarks: vary one memory parameter at a time.
# # ------------------------------------------------------------
#
# for predictor in config.PREDICTORTYPES:
#     # All default parameters.
#     default_benchmark = partial(
#         benchmarks.performance_benchmark, iterations=iterations,
#         predictortype=predictor, scenarios=benchmark_scenarios[predictor],
#         # Default memory parameters.
#         size=memsize, bw=0.1, random_fraction=0.0, temperature=0.01,
#         batching_size=256, distances="both",
#     )
#
#     # baseline: fully random memory.
#     experiment_definitions[f'benchmarks/baseline/{predictor}'] = partial(
#         default_benchmark, random_fraction=1.0)
#
#     # batching size
#     for batchsize in np.logspace(np.log10(100), np.log10(4000), 10).astype(int):
#         _path = f'benchmarks/batchsize/{predictor}/{batchsize}'
#         experiment_definitions[_path] = partial(
#             default_benchmark, batching_size=batchsize)
#
#     # bandwidth
#     for bandwidth in np.logspace(-2, 1, 10):
#         _path = f'benchmarks/bandwidth/{predictor}/{bandwidth:.2f}'
#         experiment_definitions[_path] = partial(
#             default_benchmark, bw=bandwidth)
#
#     # temperature
#     for temp in [1e-6, 1e-4, 1e-3, 1e-2, 1, 1e2, 1e4, 1e6]:
#         _path = f'benchmarks/temperature/{predictor}/{temp:.0e}'
#         experiment_definitions[_path] = partial(
#             default_benchmark, temperature=temp)
#
#     # distances
#     for distance in ("input", "output", "both"):
#         _path = f'benchmarks/distances/{predictor}/{distance}'
#         experiment_definitions[_path] = partial(
#             default_benchmark, distances=distance)


# Timing: 1M samples with 1 or 10 workers and different batchsizes.
# -----------------------------------------------------------------

# samplesize = 1000000  # Benchmark with 1 M samples
# for predictor in config.PREDICTORTYPES:
#     for batchsize in [100, 256, 1000, 4000]:
#         for workers in [1, 10]:
#             key = f'timing/{predictor}/{batchsize}-{workers}'
#             func = partial(benchmarks.time_benchmark,
#                            predictortype=predictor,
#                            memsize=memsize,
#                            scenario=benchmark_scenarios[predictor],
#                            workers=workers, iterations=iterations,
#                            batchsize=batchsize, samplesize=samplesize)
#             experiment_definitions[key] = func


# Second experiment group to tune hyperparameters.
# =================================================

tuning_group = "ns3-tuning"
tuning_experiments = {
    'workload': partial(tuning.ns3_tune, predictortype="workload"),
    'transtime': partial(tuning.ns3_tune, predictortype="transtime"),
}


# Put the definitions in the utilities for running them and proving a CLI.
# ========================================================================

ns3_experiments = eh.framework.ParametrizedExperiments(
    experiment_group, experiment_definitions,
    configcls=config.Ns3ExperimentConfig,
    cli_help="Run ns-3 experiments.",
)
ns3_tuning = eh.framework.ParametrizedExperiments(
    tuning_group, tuning_experiments,
    configcls=config.Ns3ExperimentConfig,
    cli_help="Tune ns-3 model hyperparameters.",
)
