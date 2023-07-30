"""Main simulation evaluation via data replay."""

from ..config import Ns3ExperimentConfig, PREDICTORTYPE
from .ns3_replay_helper import TranstimeDataReplay, WorkloadDataReplay


def evaluate(*, config, predictortype: PREDICTORTYPE, scenario, memorycls):
    """Run selected evaluation scenarios with selected memories."""
    config = Ns3ExperimentConfig.with_updates(config)
    replaycls = _get_replaycls(predictortype)

    try:
        mem = memorycls(config=config)
    except TypeError:
        # Not all memories need the config.
        mem = memorycls()

    replay = replaycls(
        scenario,
        memory=mem,
        config=config,
        train_runs=config.train_runs,
        eval_runs=config.eval_runs,
    )
    replay.run()


def _get_replaycls(predictortype):
    """Return framework, and memory class."""
    if predictortype == "workload":
        return WorkloadDataReplay
    if predictortype == "transtime":
        return TranstimeDataReplay
    raise ValueError("Unknown predictor type.")
