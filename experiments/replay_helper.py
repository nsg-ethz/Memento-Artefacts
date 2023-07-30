"""Utilities to help with data replay."""

import abc
import logging
from copy import deepcopy
from datetime import datetime as dt

import numpy as np
import pandas as pd
import experiment_helpers as eh
from experiment_helpers.data import Path
from experiment_helpers.typing import (Any, Dict, Iterable, Literal, Optional,
                                       PathType, Sequence, Tuple, TypedDict,
                                       TypeVar, Union)

from memento.bases import MemoryBase
from memento.models import Memento, MultiMemory
from memento.utils import (ArrayLike, StrictDataDict, merge_datadicts,
                           sample_diff)

PREDICTOR = TypeVar("PREDICTOR")
EVALDATA = TypeVar("EVALDATA")


class INPUTDATA(TypedDict):
    """A typed dict for input data annotations."""
    x: Optional[ArrayLike]
    y: Optional[ArrayLike]
    yhat: Optional[ArrayLike]
    ids: Optional[ArrayLike]
    label: Optional[str]


class ReplayFramework(abc.ABC):
    """Reusable retraining framework using a replay memory.

    At each ieration, load data input and evaluation data.
    Update the memory with the input , and test whether to retrain the predictor.
    Afterwards evaluate the current (or new) predictor on evaluation data.

    Some methods (like data loading) are left abstract and must be implemented
    by the experiment-specific subclasses. The toher methods have resonable
    defaults but may be overriden.

    The defaults are tailored to using Memento as part of a MultiMemory class
    (with Memento as the first memory) to evaluate coverage change etc.

    If `checkpoints` is True, the framework will save checkpoints after each
    iteration and resume from a checkpoint if one exists. Useful for
    long-running experiments.
    """

    # Store compressed results.
    RESULTFILENAME = "results.csv.gz"
    STATFILENAME = "stats.csv.gz"
    PREDICTORPATHNAME = "predictor"  # Might be a dir, no extension.
    CHECKPOINT = "checkpoint.pickle"  # No zipping to save time.

    ALLOWED_TRAIN_METRICS = ["coverage", "count"]

    # Init and main loop.
    # ===================

    def __init__(self,
                 memory: MemoryBase = None,
                 predictor: Optional[PREDICTOR] = None,
                 train_interval: Union[int, Literal["final"]] = 1,
                 train_threshold: Union[int, float] = 0.0,
                 train_metric: str = "coverage",
                 multi_mem_filter: Optional[Sequence[int]] = None,
                 out: Optional[PathType] = ".",
                 checkpoints: bool = True):
        # Basic parameters.
        if memory is None:
            raise TypeError("Memory must not be None.")
        self.memory = memory
        self.initial_predictor = predictor
        self.predictor = predictor

        if isinstance(train_interval, int):
            self.train_interval = train_interval
        elif train_interval == "final":
            # pylint: disable=assignment-from-none
            total = self.total_iterations()
            if total is None:
                raise ValueError("`train interval` is `final` but total number"
                                 " of iterations cannot be determined.")
            self.train_interval = total
        else:
            raise ValueError("Unknown value of `train_iterations`.")
        assert train_metric in self.ALLOWED_TRAIN_METRICS, \
            f"Retraining decision metric `{train_metric}` unknown."
        self.train_metric = train_metric
        if self.train_metric == "coverage":
            assert isinstance(train_threshold, float)
        if self.train_metric == "count":
            assert isinstance(train_threshold, int)
        self.train_threshold = train_threshold

        self.multi_mem_filter = multi_mem_filter
        self.checkpoints = checkpoints

        # Initial values.
        self.starting_iteration = 0
        self.results = None
        self.stats = None
        self.last_train_data: StrictDataDict = {}

        # Output files.
        if out is None:
            self.out = self.evalfile = self.statfile = self.checkpointfile = \
                self.predictorpath = None
        else:
            self.out = Path(out)
            self.resultfile = self.out / self.RESULTFILENAME
            self.statfile = self.out / self.STATFILENAME
            self.predictorpath = self.out / self.PREDICTORPATHNAME
            self.checkpointfile = self.out / self.CHECKPOINT

            # Load checkpoint (if it exists and is enabled.)
            self.load_state()

        # Provide memory with predictor (whether from args or file).
        if self.predictor is not None:
            self._update_mem_predictor()

    def run(self):
        """Run the retraining loop."""
        total = self.total_iterations()  # pylint: disable=assignment-from-none
        total = str(total) if total is not None else "?"

        for iteration, (input_data, eval_data) in enumerate(
                self.load_data(self.starting_iteration),
                self.starting_iteration
        ):
            logging.info("Iteration %i (of %s).", iteration+1, total)

            logging.debug("Update memory.")
            _ts = self._now()
            self._update_memory(iteration, input_data)
            mem_update_time = self._now() - _ts
            del input_data  # Allow to garbace collect.

            logging.debug("Update sample statistics.")
            training_needed = self._update_stats(
                iteration, mem_update_time, eval_data)

            if training_needed:
                logging.debug("Train predictor.")
                self._update_predictor()
            else:
                logging.debug("No training.")

            # Evaluate predictor, if there is one.
            if self.predictor is not None:
                logging.debug("Evaluate predictor")
                self._update_results(iteration, eval_data)
            else:
                # Only warn if we expect to have a predictor.
                if self.initial_predictor or (iteration >= self.train_interval):
                    logging.warning("No predictor to evaluate.")
                else:
                    logging.debug("No predictor to evaluate.")

            # Save results if output directory is provided.
            if self.out and self.checkpoints:
                logging.debug("Saving data to %s.", self.out)
                self.save_state(iteration)

        if self.out:
            logging.debug("Saving final state.")
            self.save_state(None)
            if self.checkpoints:
                logging.debug("Removing checkpoint.")
                self.remove_checkpoint()

        return self.results, self.stats

    # Implement the following methods.
    # ================================

    @abc.abstractmethod
    def load_data(self, starting_iteration: int = 0) -> \
            Iterable[Tuple[INPUTDATA, EVALDATA]]:
        """Return iterable of input and eval data."""

    def total_iterations(self) -> Optional[int]:
        """If possible return total number of iterations.

        Only used to provide better logging output or if retrain_interval is
        "final".
        """
        return None

    @abc.abstractmethod
    def evaluate(self, predictor: PREDICTOR, eval_data: EVALDATA) \
            -> pd.DataFrame:
        """Resultframe must not contain a column "iteration"."""

    @abc.abstractmethod
    def train(self, traindata: StrictDataDict,
              last_predictor: Optional[PREDICTOR] = None) -> PREDICTOR:
        """Train and return predictor from data."""

    def save_predictor(self, predictor: PREDICTOR, path: Path):
        """Save the predictor to path.

        Overwrite if pickle is not appropriate.
        """
        logging.debug("Pickle predictor to `%s`.", path)
        eh.data.to_pickle(predictor, path)

    def load_predictor(self, path: Path) -> PREDICTOR:
        """Load a predictor from path.

        Overwrite if pickle is not appropriate.
        """
        logging.debug("Unpickle predictor from `%s`.", path)
        return eh.data.read_pickle(path)

    def predictor_exists(self, path: Path) -> bool:
        """Check whether a predictor can be loaded from path.

        Overwrite if checking whether the path exists is not enough.
        """
        return path.exists()

    # Storage.
    # ========

    def save_state(self, iteration):
        """Save results and stats. If checkpoints are enabled, save more.

        Concretely, checkpoints contain the last iteration, memory batches,
        and training batches.

        Use iteration=None to indicate that iterating is over; no checkpoint
        will be saved.
        """
        logging.debug("Saving state.")
        if self.results is not None:
            eh.data.to_csv(self.results, self.resultfile)
        if self.stats is not None:
            eh.data.to_csv(self.stats, self.statfile)
        if self.predictor is not None:
            self.save_predictor(self.predictor, self.predictorpath)

        # Save Checkpoint if still in progress.
        if (iteration is not None) and self.checkpoints:
            logging.debug("Saving checkpoint `%s`.", self.checkpointfile)
            eh.data.to_pickle(
                (iteration, self.memory.data, self.last_train_data),
                self.checkpointfile
            )

    def load_state(self):
        """Load data from checkpoint."""
        # Load Checkpoint.
        if self.checkpoints and self.checkpointfile.is_file():

            logging.debug("Loading checkpoint state.")
            if self.statfile.is_file():
                self.stats = eh.data.read_csv(self.statfile)
            if self.resultfile.is_file():
                self.results = eh.data.read_csv(self.resultfile)
            if self.predictor_exists(self.predictorpath):
                self.predictor = self.load_predictor(self.predictorpath)
                self._update_mem_predictor()

            _it, _mem, _train = eh.data.read_pickle(self.checkpointfile)
            self.starting_iteration = _it + 1  # Iteration _after_ checkpoint.
            self.memory.data = _mem
            self.last_train_data = _train
        elif self.checkpoints:
            logging.debug("No previous checkpoint.")
        else:
            logging.debug("Checkpoints disabled.")

    def remove_checkpoint(self):
        """Remove checkpoint data."""
        self.checkpointfile.unlink(missing_ok=True)

    # Helper functions.
    # =================

    def compute_stats(self, iteration, memory_update_time,
                      mem_data, eval_data) \
            -> Tuple[bool, pd.DataFrame]:  # pylint: disable=unused-argument
        """Compute samplestats, return # of new samples since last training."""
        # Check whether to retrain
        decision, decision_stats = self.retrain_needed(mem_data)
        interval_ok = ((iteration + 1) % self.train_interval) == 0
        do_train = interval_ok and decision

        return do_train, pd.DataFrame({
            "iteration": iteration,
            "insert_time": memory_update_time.total_seconds(),
            "threshold": self.train_threshold,
            "interval": self.train_interval,
            "retrain": do_train,
            **decision_stats,
        }, index=[0])

    def retrain_needed(self, _data: StrictDataDict) \
            -> Tuple[bool, Dict[str, Any]]:
        """Decide whether retraining is needed."""
        if self.train_metric == "coverage":
            try:
                coverage_increase, coverage_stats = self._compute_coverage_change(
                    new_data=_data, old_data=self.last_train_data
                )
                significant_increase = coverage_increase >= self.train_threshold
            except RuntimeError as error:
                if self.train_threshold == 0:
                    # If the threshold is zero, don't crash if stats cannot be
                    # computed (e.g. because the model is missing).
                    coverage_increase = np.NaN
                    coverage_stats = {}
                    significant_increase = True
                else:
                    raise RuntimeError(
                        "Density threshold specified, but cannot "
                        "compute density change."
                    ) from error

            logging.debug("Coverage increase: `%.2f`.", coverage_increase)
            logging.debug("Significant increase: `%s`.",
                          significant_increase)

            # Return decision and statistics.
            return significant_increase, coverage_stats
        elif self.train_metric == "count":
            count_diff = sample_diff(
                self._sample_diff_filter(_data),
                self._sample_diff_filter(self.last_train_data)
            )
            significant_count = count_diff >= self.train_threshold

            logging.debug("Sample count difference: `%i`.", count_diff)
            logging.debug("Significant increase: `%s`.", significant_count)

            # Return decision and statistics.
            return significant_count, {
                "sample_count_difference": count_diff,
            }
        else:
            raise ValueError(
                f"Retrain decision metric `{self.train_metric}` unknown")

    def _sample_diff_filter(self, _data: StrictDataDict) -> StrictDataDict:
        """Preprocess data dict to e.g. ignore fields for diff."""
        return _data  # Replace in subclass if needed.

    def _update_stats(self, iteration, memory_update_time, eval_data):
        mem_data = self._comparison_data(self.memory.get())
        do_train, stats = self.compute_stats(
            iteration, memory_update_time, mem_data, eval_data)
        self.stats = self._concat(self.stats, stats)  # Append to previous.
        return do_train

    def _update_results(self, iteration, eval_data):
        results = self.evaluate(self.predictor, eval_data)
        results['iteration'] = iteration
        self.results = (
            self._concat(self.results, results)
            # Sometimes, concatenation removes e.g. category types, so cast.
            .astype(results.dtypes)
        )

    def _update_memory(self, iteration: int,  # pylint: disable=unused-argument
                       _data: StrictDataDict):
        """Insert samples. Update method if preprocessing is needed."""
        self.memory.insert(_data)

    def _update_mem_predictor(self):
        if callable(getattr(self.memory, "update_predictor", None)):
            logging.debug("Updating memory predictor.")
            self.memory.update_predictor(self.predictor)
            # Make sure to crash if anything goes wrong with the predictor now.
            self.memory.require_predictor = True
        else:
            logging.debug("Memory does not use a predictor.")

    def _update_predictor(self):
        """If the memory uses a predictor, update it."""
        self.last_train_data = deepcopy(self.memory.get())
        self.predictor = self.train(self.last_train_data,
                                    last_predictor=self.predictor)
        self._update_mem_predictor()

    def _concat(self, old: Optional[pd.DataFrame], new: pd.DataFrame):
        if old is None:
            return new
        return pd.concat([old, new], axis=0, ignore_index=True)

    def _comparison_data(self,
                         data: Union[StrictDataDict, Tuple[StrictDataDict]]):
        """Select comparison data.

        If batches are a tuple, e.g. as returned by a `MultiMemory`,
        follow `self.multi_mem_filter` to select only a subset of them,
        in particular if only some memories are relevant for comparison.
        """
        if not isinstance(data, tuple):
            logging.debug("Comparison set: Only a single memory.")
            return data
        logging.debug(
            "Comparison set: `%s` memories available.", len(data))
        if self.multi_mem_filter:
            logging.debug("Comparison set: Selecting indices `%s`.",
                          self.multi_mem_filter)
            selected = tuple(data[index] for index in self.multi_mem_filter)
        else:
            selected = data
        return merge_datadicts(selected)

    def _compute_coverage_change(self, *,
                                 new_data: StrictDataDict,
                                 old_data: StrictDataDict) -> \
            Tuple[float, Dict]:  # Density change and extra info
        """Use Memento to compute change in coverage."""
        logging.debug("Computing density change.")
        # Find memory to use.
        mem = None
        if isinstance(self.memory, Memento):
            mem = self.memory
        elif isinstance(self.memory, MultiMemory):
            for _mem in self.memory.memories:
                if isinstance(_mem, Memento):
                    mem = _mem
                    break
        if mem is None:
            raise RuntimeError("No Memento found, coverage change disabled!")

        if not old_data:
            logging.debug(r"No previous data, 100% change.")
            return 1.0, {}  # No extra info to add to stats.

        return mem.coverage_change(
            old_data, new_data, return_stats=True, return_densities=True)

    @staticmethod
    def _now():
        """Retur time. As method to be easier to mock."""
        return dt.now()
