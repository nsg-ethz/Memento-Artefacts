"""User config file.

All user and machine-specific configuration goes in here.

./run.py automatically loads this file to update the experiment config,
but a config can also be passed explicitly using the `--config <path>` option.
"""
# pylint: disable=too-few-public-methods

# import logging
from experiment_helpers.config import BaseConfig
from experiment_helpers.data import Path

current_dir = Path(__file__).parent.absolute()


class Config(BaseConfig):
    """User config."""

    # All results are stored here.
    output_directory = Path("./results")

    # Downloaded Puffer data is stored here.
    data_directory = Path("./data")

    # For ns-3 simulations, use the pre-run experiments.
    ns3_data_directory = current_dir / "ns3-simulation/results"

    # If you are using SLURM, you can override parameters below.
    # slurm_log_dir: Optional[PathType] = None
    # slurm_cpus: int = 1
    # slurm_gpus: int = 0
    # slurm_mem: int = 16
    # slurm_constraints: str = ""
    # slurm_exclude: str = ""
    # slurm_simultaneous_jobs: Optional[int] = None
    # slurm_max_array_size: Optional[int] = 1001  # Default SLURM limit.
