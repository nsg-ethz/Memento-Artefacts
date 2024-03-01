"""Global experiment config.

This config is extended in the relevant sub-directories.
"""
# pylint: disable=too-few-public-methods

from experiment_helpers.data import Path

from experiment_helpers.config import BaseConfig


class ExperimentConfig(BaseConfig):
    """Config defaults for all experiments."""

    def daystr(self, dtobj) -> str:  # pylint:disable=invalid-name
        """Consistent formatting for day object."""
        return dtobj.strftime(self.datefmt)

    # Slurm resources (only considerd with --slurm flag).
    slurm_cpus = 1
    slurm_mem = 32
    slurm_gpus = 1

    output_directory = Path("results")
    data_directory = Path("data")
