"""Memory base class."""
# pylint: disable=invalid-name,too-few-public-methods
# We have a lot of x and y here, because it would just be cumbersome otherwise.
# Also, some of our mixins only have one method (intentionally).

import abc
import logging
from typing import List, Optional, Sequence, Union

import numpy as np

from . import utils
from .utils import ArrayLike, DataDict, StrictDataDict

logger = logging.getLogger("memory.models")


# Memory base classes.
# ====================

class MemoryBase(abc.ABC):
    """Replay memory base class.

    This base class implements the framwork to handle samples, concretely:
    - inserting and getting,
    - sorting,
    - batching,
    - and updating labels.

    It does not implement the actual sample selection process, which needs to
    be implemented in subclasses.

    # Memory size

    May be infinite, although that's not really realistic.
    For debugging I guess.

    # Sample selection

    If the added samples exceed memory size, or if fewer samples are requested
    than in memory, `select` is called to determine which data to
    return. This method must be implemented by the subclass.

    The main methods to interact with the memory are `insert` and
    `get` which take arbitrary keywords to support different data
    layouts and labels. For example, using input and output data and storing
    them using them with the keys `x` and `y` respectively:

    ```
    input = np.array(...)
    output = np.array(...)
    mem.insert(x=input, y=output)
    mem.get()
    > dict(x=..., y=...)
    ```

    You can store arrays under arbitrary keys, you are not bound to `x` and `y`.

    # Batching

    By default, samples are separated into batches of size `batching_size`.
    If `batching_trim` is set to true, the final samples will be discarded if
    they are not enough to fill a batch.
    If `batching_sort` is not False, it should be a sequence of strings
    corresponding to keys in your sample data. The samples will be sorted
    according to the first key, then the second in the case of ties, and so on.
    If you want to raise an exception if data does not contain one of the keys
    required to sort it, set `batching_sort_skip_missing` to False.

    # Auto-predictions

    Includes utilities to attach a predictor to the memory to make it easily
    accessable for advanced selection strategies and to optionally automate
    getting predictions for inserted samples or when the predictor is updated.
    You can control whether to re-compute predictions on sample insert and
    predictor update with `predict_on_insert` and `predict_on_update` 
    respectively.

    As long as `self.predictor` is None, nothing will be done unless
    `require_predictor` is set to true, in which case a missing predictor will
    raise a RuntimeError when predictions are needed.

    We assume that the predictor can be called as a function that takes
    data['x'] as input and returns an array, which will be saved under
    data['y_hat], where "data" are the sample batches in the memory.
    Overwrite the prediction methods to change this behavior.
    """

    def __init__(self,
                 size: Optional[int] = None,
                 # Predictor settings.
                 predictor=None,
                 require_predictor: bool = False,
                 predict_on_insert: bool = True,
                 predict_on_update: bool = True,
                 # Batching settings.
                 batching_size: Optional[int] = None,
                 batching_trim: bool = True,
                 batching_sort: Union[Sequence[str], bool] = False,
                 batching_sort_skip_missing: bool = True,
                 ):
        """Initialize Memory."""
        self.size = size
        self.data: StrictDataDict = {}

        # Predictor settings.
        self.require_predictor = require_predictor
        self.predict_on_insert = predict_on_insert
        self.predict_on_update = predict_on_update
        if predictor is not None:
            self.update_predictor(predictor)
        else:
            self.predictor = None

        # Batching settings.
        self.batching_size = batching_size
        self.batching_trim = batching_trim
        self.batching_sort = batching_sort
        self.batching_sort_skip_missing = batching_sort_skip_missing

    def insert(self, *datadicts: DataDict, **samples: ArrayLike):
        """Insert data into memory.

        Data can be provided either as (a sequence of) dictionaries, or
        as keyword arguments of array-like data. They will be merged into a
        single dataset. The following calls are equivalent:

        ```
        memory.insert(x=[1, 2], y=[3, 4])
        memory.insert(dict(x=[1], y=[3]), x=[2], y=[4])
        memory.insert(dict(x=[1], y=[4]), dict(x=[3], y=[4]))
        memory.insert(dict(x=1, y=4), dict(x=3, y=4))
        """
        # Merge inputs, then call `insert_datadict`.
        all_datadicts = [utils.check_datadict(_d)
                         for _d in [*datadicts, samples]]
        new_data = utils.merge_datadicts(all_datadicts)
        self.insert_datadict(new_data)

    def insert_datadict(self, datadict: StrictDataDict):
        """Insert a single datadict into memory.

        Called by `insert` after merging data. If you want to preprocess
        the data before it is inserted into memory, it is easier to subclass
        this method than to subclass `insert` itself, because `insert` accepts
        multiple different input sources. This method is alwasy called with
        a single datadict.
        """
        new_len = utils.sample_count(datadict)
        current_len = utils.sample_count(self.data)
        total_count = new_len + current_len
        logger.debug("New samples:     %s", new_len)
        logger.debug("Current samples: %s", current_len)

        if self.predict_on_insert:
            if self.predictor is not None:
                # We have a predcitor.
                logger.debug("Updating predictions.")
                datadict = self.add_predictions(datadict)
            elif not self.require_predictor:
                # We don't have a predictor, but that's ok.
                logger.debug("No predictor available.")
            else:
                # We don't have a predictor and that's not ok.
                logger.critical("No predictor available.")
                raise RuntimeError("Cannot predict without predictor.")

        if ((self.size is None) or (total_count <= self.size)):
            logger.debug("Enough space for all samples, no selection.")
            self.data = utils.merge_datadicts([self.data, datadict])
        else:
            logger.debug("Selecting samples.")
            self.data = self.select(self.size, self.data, datadict)

    def get(self, max_samples: Optional[int] = None) -> StrictDataDict:
        """Return samples, optionally only a limited number."""
        if ((max_samples is None) or
                (utils.sample_count(self.data) <= max_samples)):
            logger.debug("Returning all batches.")
            return self.data
        logger.debug("Returning selected batches.")
        return self.select(max_samples, self.data, {})

    @abc.abstractmethod
    def select(self, max_samples: int,
               current_data: Optional[StrictDataDict],
               new_data: Optional[StrictDataDict]) -> StrictDataDict:
        """Select batches out of self.batches and new_batches.

        Together, the batches must contain at most `max_samples` samples.

        `new_batches` may be an empty list. This argument is only used when
        adding new batches to a finite memory, to allow implementations to
        distinguish between old (in-memory) batches and new batches.

        If there is no distinctions, feel free to use merge both, e.g. via
        `merged = utils.merge_datadicts([current_data, new_data])`.
        """

    # Predictor methods.
    # ==================

    def update_predictor(self, predictor):
        """Update the internal predictor and update predictions if required."""
        logger.debug("Updating predictor.")
        self.predictor = predictor
        if self.predict_on_update and self.data:
            logger.debug("Updating predictions.")
            self.data = self.add_predictions(self.data)

    def add_predictions(self, data: StrictDataDict) -> StrictDataDict:
        """Add predictions to data dict. Override to change keys."""
        updated = data.copy()
        try:
            updated['yhat'] = np.array(self.predict(data['x']))
        except AttributeError as error:
            raise AttributeError(
                "Your predictor has no `predict` method. Please provide this "
                "method or override `memory.predict()`.") from error
        return updated

    def predict(self, x: np.ndarray) -> ArrayLike:
        """Get predictions for x. Override for custom methods."""
        return self.predictor(x)  # type: ignore

    # Batch methods.
    # ==============

    def batch(self, data: StrictDataDict) -> List[StrictDataDict]:
        """Split data into batches."""
        if not data:
            logger.debug("No data to batch.")
            return []
        if self.batching_sort:
            logger.debug("Sorting data.")
            data = self.sort(data)
        else:
            logger.debug("Sorting disabled.")

        total = utils.sample_count(data)
        logger.debug("Splitting `%s` samples into batches.", total)
        batches = []
        n_fit = total // self.batching_size
        last_fit = n_fit * self.batching_size
        if last_fit > 0:
            splits = np.split(np.arange(last_fit), n_fit)
            batches.extend([{k: v[idx] for k, v in data.items()}
                            for idx in splits])
        diff = total-last_fit
        if diff and not self.batching_trim:
            logger.warning("Final batch only has size `%i`.", diff)
            batches.append({k: v[last_fit:] for k, v in data.items()})
        elif diff and self.batching_trim:
            logger.debug("`%i` samples removed by trimming.", diff)

        return batches

    def sort(self, data: StrictDataDict) -> StrictDataDict:
        """Sort data using lexsort, override for custom sort.

        Override `sort_args()` to keep lexsort but modify order or similar.
        """
        sort_index = np.lexsort(self.sort_args(data))
        return {k: a[sort_index] for k, a in data.items()}

    def sort_args(self, data: StrictDataDict) -> List[np.ndarray]:
        """Return arguments for lexicographic search from data.

        `batching_sort` determines the order of sort, with the first being the
        most important. Each value in `batching_sort_columns` must
        correspond to a key in the data dict.

        If a key is missing, an exception is raised, unless
        `batching_sort_skip_missing` is True.

        If the corresponding data is 2d, it is sorted first by argmax of the
        absolute values, then corresponding value.
        For probability distributions, this corresponds to sorting by mode
        with decreasing probability.

        Implementation details:
        Numpy lexsort sorts by the last column first, so they need to be
        provided from least to most important.
        To efficiently use lexsort, we creat only views to the individual
        array columns, to avoid costly allocation of new arrays.
        """
        try:
            # Reverse, as the last argument will be sorted by first.
            keys = self.batching_sort[::-1]  # type: ignore
        except TypeError as error:
            raise TypeError("`self.batching_sort` must be a sequence of data "
                            f"keys but got `{self.batching_sort}`") from error
        columns = []
        for key in keys:
            try:
                array = data[key]
            except KeyError as error:
                msg = "Cannot sort by `%s` (missing)."
                if self.batching_sort_skip_missing:
                    logger.warning(msg, key)
                    continue
                logger.critical(msg, key)
                raise KeyError(msg % key) from error
            if array.ndim == 1:
                columns.append(array)  # Only one column
            elif array.ndim == 2:
                _argmax = np.argmax(np.abs(array), axis=1)
                _max = array[np.arange(len(array)), _argmax] * -1
                # Sort first by _which_ element is max, then by value.
                # * -1 to go from high probability to low.
                # Again: most important columns need to be placed last.
                columns.extend([_max, _argmax])
            else:
                msg = "Sorting is currently not supported for more than 2d!"
                logger.critical(msg)
                raise RuntimeError(msg)
        if not columns:
            raise RuntimeError(
                f"Failed to sort! Sort keys: `{self.batching_sort}`; "
                f"data keys: `{list(data)}`")
        return columns
