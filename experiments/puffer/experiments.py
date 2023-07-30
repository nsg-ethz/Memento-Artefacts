"""All Puffer experiments.

This file defines experiments in the form:

```
name = "puffer"
defintions = {
    "<path>": function(config)
}
```

That is, each experiment is identified by the path where its outputs are
stored, and is a function that takes a (user) config as argument.

When running these experiments with the `experiment_helpers`,
outputs are stored in `config.output_directory / puffer / <path>`.

Implementation note:
Below, we use functools.partial to prepare the functions. It allows to
partially set some function arguments (hence the name), so we can easily
parametrize one experiment function in different ways to define a range
of experiments.
"""
# pylint: disable=invalid-name

from functools import partial
from typing import Callable, Dict

import pandas as pd
from experiment_helpers.framework import ParametrizedExperiments

from . import config
from .implementation import (analysis, data, deployment, memory, replay,
                             selection_analysis)

replay_group = "puffer-replay"
replay_exps: Dict[str, Callable] = {}

# Memory size of 1M, like puffer.
memsize = 1000000
# Limit how many samples we process at once.
# The majority of days has less then 2M samples, so most days we can process
# in one go. However, there are a few days with a _lot_ of samples (up to 9M).
# While not an issue per-se, this causes Memento to need a lot of memory.
# As other people also need to server, we can limit how much we process :D
max_insert = 2000000

# Default settings we use for Puffer Memento.
default_memento = partial(
    memory.PufferMemory,
    size=memsize, batching_size=256,
    bw=0.1, temperature=0.01,
    insert_chunksize=max_insert,
    random_fraction=0.0, random_forget=0.0,
)

default_alternatives = {
    "confidence":
    partial(memory.PufferConfidence, size=memsize, temperature=0.1),
    "loss":
    partial(memory.PufferLoss, size=memsize, temperature=0.1),
    "loss_batched":
    partial(memory.PufferLoss, size=memsize, temperature=0.1,
            batching_size=256),
    "classcounts":
    partial(memory.PufferClassCounts, size=memsize, temperature=0.01),
    "stalls":
    partial(memory.PufferStalled, size=memsize, temperature=0.01),
}

# Default retraining threshold.
default_threshold = 0.1  # 10%

_today = pd.to_datetime("today").strftime("%Y-%m-%d")
data_first_and_last = {
    # Use data from 2021 for experiments.
    'comparison': ("2021-01-01", "2022-06-01"),
    # Evaluate long-term results until cutoff before submission.
    'eval': ("2020-04-09", "2023-07-17"),
    # Current memento deployment, including latest data.
    'deployment': ("2022-05-01", _today),
}
data_ranges = {
    key: pd.date_range(start_day, end_day, freq='D')
    for key, (start_day, end_day) in data_first_and_last.items()
}

# Data download.
# ==============
# We put them in a different group of commands, because the downloads need
# to be run independently of the other experiments.
# Running both at the same time can lead to experiments being run with
# incomplete data.

download_group = "puffer-download"
download_exps: Dict[str, Callable] = {}

# Note: The path here does not matter for outputs, as download stores the
# data into the `puffer_data_directory` specified in the config.

# Always make sure we have data for Fugu-Feb downloaded!
download_exps['fugufeb'] = partial(data.download_data, day="fugufeb")

for days in data_ranges.values():
    for day in days:
        download_exps[f"{day.strftime('%Y-%m-%d')}"] = partial(
            data.download_data, day=day)


# Data preprocessing.
# ===================
# We need a lot of data in the correct format for the TTP models.
# We can preprocess them to speed up processing at the expense of disk space.

preprocess_group = "puffer-preprocess"
preprocess_exps: Dict[str, Callable] = {}

for days in data_ranges.values():
    for day in days:
        preprocess_exps[f"{day.strftime('%Y-%m-%d')}"] = partial(
            data.preprocess_data, day=day)


# Data analysis.
# ==============

analysis_group = "puffer-analysis"
analysis_exps: Dict[str, Callable] = {}

for day in data_ranges['eval']:
    analysis_exps[f"{day.strftime('%Y-%m-%d')}"] = partial(
        analysis.analyze_day, day=day)


# Comparison of different parameters.
# ===================================

# Start and end day of comparison runs.
# Each half a year. The first day is only used for training, the last day
# only for eval, so it's ok if runs overlap on that.
comparison_runs = [
    ("2021-01-01", "2021-07-01"),
    ("2021-07-01", "2022-01-01"),
    ("2022-01-01", "2022-06-01"),
]

for start, end in comparison_runs:
    run = f"{start}_{end}"
    # Data stats like streamtime for the comparison run.
    # --------------------------------------------------
    replay_exps[f'stats/{run}'] = partial(
        replay.data_overview, startday=start, endday=end)

    # pylint: disable=dangerous-default-value
    # (we only read from the dict).

    def _parametrize_replay(config_overrides=None, replay_kwargs=None,
                            memcls=None, **mem_kwargs):
        """The replay function takes a lot of parameters.

        This function is a helper to parametrize it.
        """
        if memcls is None:
            memcls = partial(default_memento, **mem_kwargs)
        else:
            memcls = partial(memcls, **mem_kwargs)
        # By default, we retrain once per week.
        # We need to retrain to access the memory quality, but retraining
        # daily is unnecessarily expensive to compute.
        _rkws = dict(
            retrain_from="fugu_feb",  # Speed up training.
            train_threshold=0.0, train_interval=7
        )
        if replay_kwargs:
            _rkws.update(replay_kwargs)
        replaycls = partial(replay.PufferDataReplay, **_rkws)

        return partial(
            replay.evaluate,
            startday=start, endday=end,  # pylint: disable=cell-var-from-loop
            replaycls=replaycls, memorycls=memcls,
            config_overrides=config_overrides,
        )

    # Default parameters as baseline.
    # -------------------------------

    # Default parameters.
    replay_exps[f'comparison/baseline/default/{run}'] = _parametrize_replay()
    replay_exps[f'comparison/baseline/random/{run}'] = _parametrize_replay(
        random_fraction=1.0,  # totally random mem.
        random_forget=0.1,   # with same forgetting rate as puffer.
    )
    # Memento (T=0), i.e. no noise rejection. Only keep lowest density.
    replay_exps[f'comparison/baseline/deterministic/{run}'] = \
        _parametrize_replay(temperature=0.0)

    # Parameter sweeps.
    # -----------------

    # Different temperature settings for random scaling.
    for temp in [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1, 1e1, 1e2]:
        replay_exps[f'comparison/temperature/{temp:.0e}/{run}'] = \
            _parametrize_replay(temperature=temp)

    # Different bandwidths.
    for bw in [0.001, 0.01, 0.05, 0.08, 0.1, 0.125, 0.2, 0.5, 1, 10]:
        replay_exps[f'comparison/bandwidth/{bw:g}/{run}'] = \
            _parametrize_replay(bw=bw)

    # Different batching sizes (not too small or it takes forever).
    for bsize in [128, 256, 512, 1024, 2048]:
        replay_exps[f'comparison/batchsize/{bsize}/{run}'] = \
            _parametrize_replay(batching_size=bsize)

    # Optional: Only consider input or output space.
    # Both are beneficial; not shown in paper; uncomment to run.
    # for distances in ("both", "input", "output"):
    #     replay_exps[f'comparison/subspace/{distances}/{run}'] = \
    #         _parametrize_replay(distances=distances)

    # Ablation: compare smart Memory to a larger, but naive, memory.
    replay_exps[f'comparison/memsize/double/{run}'] = _parametrize_replay(
        size=2*memsize,  # double the size.
        random_fraction=1.0,  # totally random mem.
        random_forget=0.1,   # with same forgetting rate as puffer.
    )

    # Test alternative selection methods than density.
    # These methods also support different temperatures to control randomness.
    altexp = "comparison/metric"
    for alt_name in ["loss", "stalls", "confidence", "classcounts"]:
        alt_cls = default_alternatives[alt_name]

        # For stalls we need to load them; save the overhead otherwise.
        replay_kws = dict(load_stalls=True) if alt_name == "stalls" else {}

        for temp in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:
            replay_exps[f'{altexp}/{alt_name}/{temp:.0e}/{run}'] = \
                _parametrize_replay(memcls=alt_cls, temperature=temp,
                                    replay_kwargs=replay_kws)

    # For the loss-based alternative, some additional datapoints are helpful.
    for temp in [1e2, 1e3]:
        replay_exps[f'{altexp}/loss/{temp:.0e}/{run}'] = \
            _parametrize_replay(memcls=default_alternatives['loss'],
                                temperature=temp)

    # compare training decision metrics.
    # ----------------------------------
    # coverage-based (Memento) vs loss-based.

    for decision in ["coverage", "loss"]:
        for threshold in [0.05, 0.1, 0.2]:
            replay_exps["comparison/train-decision/"
                        f"{decision}/{threshold:.2f}/{run}"] = \
                _parametrize_replay(replay_kwargs=dict(
                    train_interval=1,  # Allow retraining every day.
                    # But don't automatically retrain -> decide each day.
                    train_threshold=threshold,
                    train_metric=decision,
                ))

    # compare with and without JTT for training and MatchMaker for predictions.
    # -------------------------------------------------------------------------

    # To avoid an combinatorial number of experiments, we only test a few
    # a few combinations:
    # - Only JTT (what if we only try to improve training instead of samples?)
    # - JTT + Memento (what if we do both?)
    # - JTT + Memento + Matchmaker (what if we also improve predictions?)

    # All other possible experiments are outlined below. Uncomment if you want
    # to run them.

    # No Upscaling.
    # JTT without upscaling is just training from random samples, we can use
    # the baseline result for that.
    # JTT (no upscaling) & Memento is just default Memento, use that as well.
    # JTT (no upscaling) & Memento & MatchMaker: Default Memento + MatchMaker.
    replay_exps[
        f"combinations/memento-matchmaker/{run}"
    ] = _parametrize_replay(replay_kwargs=dict(matchmaker_predictors=7))

    # Uncomment to test only MatchMaker on its own.
    # replay_exps[
    #     f"combinations/matchmaker/{run}"
    # ] = _parametrize_replay(
    #     random_fraction=1.0,  # totally random mem.
    #     random_forget=0.1,   # with same forgetting rate as puffer.
    #     replay_kwargs=dict(matchmaker_predictors=7),
    # )

    # With JTT upscaling.
    for upscale in [2., 3., 5., 10., 20., 50., 100.]:
        # JTT only (better training with same samples).
        replay_exps[f"combinations/jtt/{upscale:g}/{run}"] = \
            _parametrize_replay(
                random_fraction=1.0,  # totally random mem.
                random_forget=0.1,   # with same forgetting rate as puffer.
                replay_kwargs=dict(jtt_upscale=upscale),
        )
        # JTT & Memento
        replay_exps[
            f"combinations/memento-jtt/{upscale:g}/{run}"
        ] = _parametrize_replay(replay_kwargs=dict(jtt_upscale=upscale))
        # JTT & Memento & MatchMaker
        replay_exps[
            f"combinations/memento-matchmaker-jtt/{upscale:g}/{run}"
        ] = _parametrize_replay(
            replay_kwargs=dict(jtt_upscale=upscale, matchmaker_predictors=7),
        )

        # Uncomment to test only JTT & MatchMaker without Memento.
        # replay_exps[
        #     f"combinations/matchmaker-jtt/{upscale:g}/{run}"
        # ] = _parametrize_replay(
        #     random_fraction=1.0,  # totally random mem.
        #     random_forget=0.1,   # with same forgetting rate as puffer.
        #     replay_kwargs=dict(jtt_upscale=upscale, matchmaker_predictors=7),
        # )

        # JTT uses a first training round to decide what to upscale.
        # Alterantively, we could e.g. upscale all sessions that struggle.
        # First results were not promising, so it's commented out.
        # Uncomment if you want to try it out.
        # replay_exps[f"combinations/stalls/{upscale:g}/{run}"] = \
        #     _parametrize_replay(
        #         random_fraction=1.0,  # totally random mem.
        #         random_forget=0.1,   # with same forgetting rate as puffer.
        #         replay_kwargs=dict(stall_upscale=upscale),
        # )


# Analyze how well future samples are covered by the memory samples.
# ==================================================================
# Collect data over three weeks, then freeze the memory. For the rest of the
# year, only evaluate.

freeze_replayclass = partial(
    replay.PufferDataReplay,
    retrain_from="fugu_feb",  # Speed up training.
    train_threshold=0.0,
    train_interval=7,
    freeze_after=21,  # Do not update memory or retrain after iteration 21.
)

freeze_replay = partial(
    replay.evaluate, replaycls=freeze_replayclass,
    startday="2022-01-01", endday="2022-12-31",
)

replay_exps["freeze/memento/2022"] = partial(
    freeze_replay, memorycls=default_memento)
random_mem = partial(default_memento, random_fraction=1, random_forget=0.1)


# Analyze the memory selection in-depth over a month.
# ===================================================

selection_exp = partial(
    selection_analysis.analyse_selection,
    # Select 5M from the start of 2022.
    n_samples=5000000, start="2022-01-01", end="2022-01-31",
    batch_memorycls=default_memento,
)

replay_exps['selection/memento'] = partial(
    selection_exp, memorycls=default_memento)
# loss _batched_ means: same batching as Memento, but loss as metric.
# loss does not need batching, but we want to ensure that the only difference
# is the selection strategy.
replay_exps['selection/loss_batched'] = partial(
    selection_exp, memorycls=default_alternatives['loss_batched'])

# Alternative metrics loss without batching or confidence (max(p)).
# Don't provide additional insights compared to loss. Uncomment to run anyways.
# replay_exps['selection/loss'] = partial(
#     selection_exp, memorycls=default_alternatives['loss'])
# replay_exps['selection/confidence'] = partial(
#     selection_exp, memorycls=default_alternatives['confidence'])


# Continual learning deployment.
# ==============================

deployment_default = default_memento
# As comparison, also use a Memento variant that does not choose randomly.
deployment_deterministic = partial(default_memento, temperature=0.0)
deployment_threshold = 0.1  # 10%


# Put this into a separate group of experiments.
deploy_group = "puffer-deployment"
deploydefs: Dict[str, Callable] = {}
deployment_start, deployment_end = data_first_and_last['deployment']
for index in range(5):  # Each index corresponds to a model horizon.
    deploydefs[f'default/{index}'] = partial(
        deployment.update, index=index,
        first_day=deployment_start, last_day=deployment_end,
        memcls=deployment_default, threshold=deployment_threshold,
        git_dir="/home/alex/puffer_fugu_variant"
    )
    deploydefs[f'deterministic/{index}'] = partial(
        deployment.update, index=index,
        first_day=deployment_start, last_day=deployment_end,
        memcls=deployment_deterministic, threshold=deployment_threshold,
        git_dir="/home/alex/puffer_fugu_variant_2"
    )

# Put the definitions in the utilities for running them and proving a CLI.
# ========================================================================
# We pass the config classes because they contain framework defaults as well.

puffer_download = ParametrizedExperiments(
    download_group, download_exps,
    configcls=config.PufferExperimentConfigNoGPU,
    cli_help="Download Puffer data.",
)
puffer_preprocess = ParametrizedExperiments(
    preprocess_group, preprocess_exps,
    configcls=config.PufferExperimentConfigNoGPU,
    cli_help="Preprocess Puffer data (optional).",
)
puffer_analysis = ParametrizedExperiments(
    analysis_group, analysis_exps,
    configcls=config.PufferExperimentConfig,
    cli_help="Analyze Puffer data and aggregate ABR performance.",
)
puffer_replay = ParametrizedExperiments(
    replay_group, replay_exps,
    configcls=config.PufferExperimentConfig,
    cli_help="Run Puffer data replay.",
)
puffer_deployment = ParametrizedExperiments(  # Note: different base config!
    deploy_group, deploydefs, configcls=config.PufferDeploymentConfig,
    cli_help="Update Puffer deployment of Memento.",
)
