"""Config specific to the puffer experiments."""

from datetime import datetime as dt
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
from experiment_helpers.data import Path

from ..config import ExperimentConfig


class PufferExperimentConfig(ExperimentConfig):
    """Config subclass for Puffer."""

    # Slurm settings -> we need enough memory.
    # We request a lot of memory, but it's really only needed for a few days
    # with tons of data. Most of days won't use this much.
    slurm_mem = 64  # GB.

    @property
    def puffer_data_directory(self):
        """Use a subdirectory of the data directory."""
        return Path(self.data_directory) / "puffer"

    def get_data_directory(self, day: dt):
        """Return the data directory for a given day."""
        return Path(self.puffer_data_directory) / self.daystr(day)

    # The day for the model used online as FUgu-Feb
    fugu_feb_day = dt(2019, 2, 2)

    @property
    def fugu_feb_dir(self):
        """Return dynamic path where Fugu-Feb should be."""
        return self.get_data_directory(self.fugu_feb_day) / self.model_pathname

    # Training settings.
    train_epochs: int = 3000     # Allow longer max training.
    train_patience: int = 50     # and use early stopping,
    train_validation_split: float = 0.2
    train_min_delta: float = 1e-6
    train_batchsize: int = 512  # and bigger batches (faster, same perf).
    train_custom_size: Optional[Tuple[int, int]] = None

    # Analysis.
    # ---------

    evaluation_mapping = {
        # public name, local name
        "linear_bba": "bba",  # baseline without model.
        "puffer_ttp_cl": "fugu",
        "puffer_ttp_20190202": "fugu-feb",
        "fugu_variant_cl3": "memento",
        "fugu_variant_cl4": "memento-deterministic",
        # Old labels before they were renamed.
        "fugu_variant_cl": "memento",
        "fugu_variant_cl2": "memento-deterministic",
    }

    memento_model_basepath = Path("./results/puffer-deployment")

    @property
    def memento_model_directories(self):
        """Path to the models we train for deployment.

        Models in this list are included in the analysis.
        """
        return {
            "memento": self.memento_model_basepath / "default",
            "memento-deterministic":
            self.memento_model_basepath / "deterministic",
        }

    chunk_playtime = 2.001  # seconds.
    # Each chunk is 2.001s playtime, Puffer wants >= 4s, i.e. at least 2 chunks.
    analysis_min_chunks = 2
    analysis_quantiles = [
        0.0, 0.001, 0.01,
        *np.linspace(0.1, 0.9, 9),
        0.99, 0.999, 1.0,
    ]

    # Download config.
    # ---------------

    download_timeout = 10.0  # seconds.
    download_poll_interval = None   # Don't wait for data.

    # Whether to prepare the inout data when downloading. Takes more space.
    preprocess_inout = False  # Mem issue for now.

    # Default filenames (for downloading and accessing)
    video_filename = "chunk_data.csv.gz"
    inout_filenames = [f"inout.{index}.npz" for index in range(5)]
    model_pathname = "model"

    base_data_url = "https://storage.googleapis.com/puffer-data-release/"
    base_model_url = "https://storage.googleapis.com/puffer-models/puffer-ttp/"

    def get_puffer_data_url(self, day: dt, kind: str, model_version: int = 1):
        """Get the URL for puffer data."""
        if kind == "model":
            # Archive of trained models are stored per day.
            archive = f"bbr-{day.strftime('%Y%m%d')}-{model_version}.tar.gz"
            return self.base_model_url + archive

        # All other data are stored under date ranges.
        nextday = day + timedelta(days=1)
        fmt = "%Y-%m-%dT11"  # format used in fugu storage
        daterange = f"{day.strftime(fmt)}_{nextday.strftime(fmt)}"

        if kind == "exp":
            # Experiment information (abr, cc).
            return f"{self.base_data_url}{daterange}/logs/expt_settings"
        if kind in ("ssim", "video_sent", "video_acked",
                    "video_size", "client_buffer"):
            return f"{self.base_data_url}{daterange}/{kind}_{daterange}.csv"
        raise ValueError(f"Unknown kind {kind}.")

    # The daily retained fugu was discontinued on 2022-10-06.
    last_fugu_day = dt(2022, 10, 6)

    # On some days, the data is corrupted and cannot be downloaded.
    # Various reasons: pipeline failure, once there was a blackout on the
    # Stanford campus, etc. The data is gone, so we don't need to throw an
    # error for every download.
    data_known_missing = (
        dt(2020, 7, 7),
        dt(2020, 7, 8),
        dt(2020, 8, 6),
        # 2021
        dt(2021, 1, 19),
        dt(2021, 3, 12),
        dt(2021, 3, 21),
        dt(2021, 6, 2),
        dt(2021, 6, 3),
        dt(2021, 6, 4),
        dt(2021, 6, 13),
        dt(2021, 6, 14),
        dt(2021, 8, 6),
        dt(2021, 8, 17),
        dt(2021, 9, 21),
        dt(2021, 10, 18),
        dt(2021, 11, 8),
        dt(2021, 11, 9),
        dt(2021, 11, 10),
        dt(2021, 11, 11),
        dt(2021, 11, 14),
        dt(2021, 11, 29),
        dt(2021, 11, 30),
        dt(2021, 12, 10),
        # 2022
        dt(2022, 1, 24),
        dt(2022, 1, 31),
        dt(2022, 2, 1),
        dt(2022, 2, 2),
        dt(2022, 2, 3),
        dt(2022, 2, 4),
        dt(2022, 2, 5),
        dt(2022, 2, 6),
        dt(2022, 2, 7),
        dt(2022, 2, 8),
        dt(2022, 2, 9),
        dt(2022, 2, 10),
        dt(2022, 2, 11),
        dt(2022, 2, 12),
        dt(2022, 2, 13),
        dt(2022, 2, 14),
        dt(2022, 2, 15),
        dt(2022, 5, 25),
        dt(2022, 6, 21),
        dt(2022, 6, 22),
        dt(2022, 6, 23),
        dt(2022, 6, 24),
        dt(2022, 6, 25),
        dt(2022, 6, 26),
        dt(2022, 7, 17),
        dt(2022, 7, 18),
        dt(2022, 8, 19),
        dt(2022, 8, 25),
        dt(2022, 9, 22),
        dt(2022, 9, 23),
        dt(2022, 9, 24),
        dt(2022, 9, 25),
        dt(2022, 9, 26),
        dt(2022, 9, 14),
        dt(2022, 9, 18),
        dt(2022, 9, 27),
        dt(2022, 9, 28),
        dt(2022, 10, 4),
        # 2023
        dt(2023, 6, 24),
        dt(2023, 6, 25),
        dt(2023, 6, 26),
        dt(2023, 6, 27),
        dt(2023, 6, 28),
        dt(2023, 6, 29),
        dt(2023, 6, 30),
        dt(2023, 7, 1),
    )


class PufferExperimentConfigNoGPU(PufferExperimentConfig):
    """set slurm gpu to 0."""
    slurm_gpus = 0
    # The CPU jobs need more memory to process days with a lot of data.
    slurm_mem = 128  # GB


class PufferDeploymentConfig(PufferExperimentConfig):
    """An extended config for deployment with extra logging."""
    logfiles = {
        'memory': "./logs/memory.log",
        'experiments': "./logs/experiment.log"
    }
    logfile_kwargs = dict(maxBytes=10 * 1024 * 1024, backupCount=5)
