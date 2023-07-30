"""Default parameters for the ns3 models and data processing."""

from typing import Literal

import numpy as np
from experiment_helpers.data import Path

from ..config import ExperimentConfig

TRAINRUNS = tuple(range(2, 22))

PREDICTORTYPES = ["workload", "transtime"]
PREDICTORTYPE = Literal["workload", "transtime"]


class Ns3ExperimentConfig(ExperimentConfig):
    """ns-3 configuration defaults."""
    random_state = 42
    multiprocessing_context = "spawn"  # Required for tensorflow.

    @property
    def ns3_data_directory(self):
        """Use a subdirectory of the data directory."""
        return Path(self.data_directory) / "ns3"

    # slurm_log_dir = ".slurmlogs"
    slurm_gpus = 0  # My cuDNN install does not work, so disable GPU usage.
    slurm_cpus = 20  # Compensate for no GPU.

    # How to use the experiment runs (tuning/eval/training).
    # Should be configured to match data.
    runs = tuple(range(22))
    tuning_runs = (0,)
    eval_runs = (1,)
    train_runs = TRAINRUNS

    # Different split for benchmarks
    benchmark_baseline_runs = TRAINRUNS[:2]
    benchmark_runs = TRAINRUNS[2:]

    # General parameters.
    memsize = 20000
    horizon = 128
    workloads = 3
    transtime_support = np.linspace(0, 0.1, 128)
    transtime_kde_bw = "silverman"  # Adaptive yet fast.

    # Tuning parameters
    tuning_iterations = 100
    tuning_train_size = 0.8
    tuning_splits = 5
    tuning_penalty = 10   # "Worst case" value for invalid results.

    # Workload model training parameters.
    workload_epochs = 200
    workload_patience = 0
    workload_validation_split = 0.0
    workload_hidden_layers = 3
    workload_hidden_units = 512
    workload_learning_batchsize = 512
    workload_learning_rate = 5.911721340393673e-05

    # Transtime model training parameters.
    transtime_epochs = 200
    transtime_patience = 50
    transtime_validation_split = 0.2
    transtime_hidden_layers = 4
    transtime_hidden_units = 362
    transtime_learning_batchsize = 128
    transtime_learning_rate = 0.38227137916547793
