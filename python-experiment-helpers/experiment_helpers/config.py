"""The base configuration class and helper functions.

All custom configurations should inherit from this class.
If they don't, features such as logging and SLURM will not work.

The config class provides a `with_updates` classmethod that allows load the
class as defaults, with user-defined updates, e.g. machine specific settings.

Finally, also defines a `ConfigT` type variable for type annotations.
"""
# pylint: disable=invalid-name

import logging
import types
from copy import copy
from os import PathLike
from typing import Any, Dict, Optional, Sequence, Type, TypeVar

from .data import Path, PathType

# Create a generic variable that can be 'Parent', or any subclass.
ConfigT = TypeVar('ConfigT', bound='BaseConfig')


class BaseConfig:  # pylint: disable=too-few-public-methods
    """Basic config class.

    Contains default configuration for all framework utilities as well
    as an method to instantiate a general config with specific updates.
    """
    # Output directory as a root for experiment results.
    output_directory: PathType = Path('./results')

    # When an experiment finishes, this file will be created with runtime info.
    # This also indicates that the experiment can be skipped in future runs.
    meta_filename: str = "experiment_helpers_metadata.json"

    # Special files when checking output directories.
    # Checkpoints present: run is not complete.
    checkpoint_files: Sequence[str] = ('*checkpoint*',)
    # These files are ignored. TODO: Deprecate.
    ignore_files: Sequence[str] = ('*.tmp',)

    # How many workers each experiment function may start.
    workers: int = 1

    # Loggings.
    loglevel: int = logging.WARNING
    logbasename: str = "experiments"
    # To support multiprocessing _and_ SLURM, exchange logs via files.
    logpickledir: PathType = Path('./.logs')
    # When using SLURM, we need to actively poll the file system for new logs.
    # We also need to poll while waiting for logs to finish (with timeout).
    logpollinterval: float = 0.1  # in seconds.
    logpolltimeout: float = 10.0  # in seconds.
    # Optionally log everything (logging.DEBUG) to files.
    logfiles: Dict[str, str] = {}  # Map logger to file.
    logfile_kwargs: Dict = dict(maxBytes=0, backupCount=0)

    @property
    def base_loglevel(self):
        """Return the base loglevel to support all handlers."""
        return self.loglevel if not self.logfiles else logging.DEBUG

    # Basic formatting styles.
    utcfmt = "%Y-%m-%dT%H:%M:%SZ"  # ISO 8601 for UTC without ms.
    datefmt = '%Y-%m-%d'

    # Randomly delay function execution when multiprocessing to avoid bursty
    # resource access.
    random_start_delay: Optional[float] = 0.5  # in seconds.

    # Multiprocessing settings; Not relevant if SLURM is used.
    multiprocessing_jobs: int = 1  # How many functions are run in parallel.
    multiprocessing_context: str = "fork"

    # SLURM, alternative to multiprocessing.
    slurm: bool = False
    slurm_shell: str = "/bin/bash"
    slurm_interpreter: str = "python"
    slurm_sbatch_cmd = "sbatch --wait"
    slurm_scancel_cmd = "scancel"
    slurm_log_dir: Optional[PathType] = None
    slurm_cpus: int = 1
    slurm_gpus: int = 0
    slurm_mem: int = 16
    slurm_constraints: str = ""
    slurm_exclude: str = ""
    slurm_time_limit: Optional[int] = None  # in minutes
    slurm_simultaneous_jobs: Optional[int] = None
    slurm_max_array_size: Optional[int] = 1001  # Default SLURM limit.

    # Slack.
    slacktoken: Optional[str] = None
    slackchannel: Optional[str] = None

    # Instantiate config with updates.
    # ================================

    @classmethod
    def with_updates(cls: Type[ConfigT],
                     updates: Any = None,
                     overrides: Optional[Dict[str, Any]] = None) -> ConfigT:
        """Instantiate the base config class with user updates and overrides.

        The following logic applies:
        - If no updates (None or empty dict), return base config instance.
        - If updates are already a subclass of cls, return directly.
        - If updates are a class (but no subclass), instantiate a new subclass
        that has user class first in resolution order, and cls second.
        - If update are a dict, instantiate cls, and update attributes from dict.

        For any other type of object, inject cls as a baseclass so that missing
        attributes default to it (as intended).

        Finally, apply optional overrides from a dict. To avoid mistakenes,
        only attributes that are already present in the config class are
        allowed.

        TODO: Overrides are limited, they cannot override anyproperties;
        setattr only works for properties that have defined a setter.
        Idea: make this more dynamic and allow arbitrary *updates, with the
        restriction that classes need to come before instances. But I am not
        fully sure how dicts would fit into this approach. Not critical atm.
        """
        # Updates.
        # --------
        if updates is None:
            config = cls()            # No updates, use base as-is.
        # Updates using user config classes.
        elif isinstance(updates, type) and issubclass(updates, cls):
            config = updates()        # Already a subclass, use user as-is.
        elif isinstance(updates, type):
            # pylint: disable=too-few-public-methods
            class MergedConfig(updates, cls):  # type: ignore
                """A merged config class. The user class takes precedence."""
            config = MergedConfig()
        # Updates using user config objects (incl. dicts).
        elif isinstance(updates, cls):
            config = updates          # Already an instance, use user as-is.
        elif isinstance(updates, dict):
            # Create a subclass from dict. Necessary to override properties.
            config = type("UserConfig", (cls,), updates)()
        else:
            # Finally, the most "magic" -- for any object, inject `cls` as
            # a baseclass, such that all missing attributes default to it.
            config = copy(updates)  # Avoid changing provided object.
            # Implementation: type() with three arugments creates a class.
            config.__class__ = type(
                "MergedConfig", (config.__class__, cls), {})

        # Overrides.
        # ----------
        if overrides is not None:
            for key, value in overrides.items():
                assert hasattr(config, key), \
                    f"The config has no attribute {key}."
                setattr(config, key, value)

        return config


def load_config_from_pyfile(path: PathLike, key="Config"):
    """Load `Config` object/class from a python file.

    This is _not_ safe, only load trusted files.

    Code taken from: https://github.com/pallets/flask/blob/41aaaf7fa07f28ef15fd1d343142e96440889a8b/src/flask/config.py#L165
    """
    target = types.ModuleType("config")
    target.__file__ = str(path)
    with open(path, mode="rb") as config_file:
        # pylint: disable=exec-used
        exec(compile(config_file.read(), path, "exec"), target.__dict__)
    return getattr(target, key)
