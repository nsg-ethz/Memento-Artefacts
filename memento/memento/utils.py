"""Utilities."""

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Union)

import numpy as np
from numpy.typing import ArrayLike

DataDict = Dict[str, ArrayLike]
StrictDataDict = Dict[str, np.ndarray]  # stricter type.


# Required for array sharing with multiprocessing.
SharedArrays = Optional[Union[Any, Sequence[Any]]]  # Currently no better type.
SHARED_ARRAYS: SharedArrays = None


class MultiProcessing():
    """Helper to provide multiprocessing utilities and initialization."""

    def __init__(self, *args,
                 workers: Optional[int] = None,
                 **kwargs):
        self.workers = workers if workers is not None else 1
        # For type ignore, see: https://github.com/python/mypy/issues/4001
        super().__init__(*args, **kwargs)  # type: ignore

    def map(self, function: Callable, jobs: Sequence,
            shared_arrays: SharedArrays = None) -> List:
        """Execution function in parallel for jobs using `map`.

        If `self.workers` is 1, the jobs are executed sequentially
        for easier debugging.

        Shared arrays cannot be passed as function arguments, and are supported
        as separate inputs to map. See https://stackoverflow.com/a/39322905
        They are available to the function via a global variable SHARED_ARRAYS.
        """
        if self.workers == 1:
            if shared_arrays is not None:
                _init(shared_arrays)
            result = [function(job) for job in jobs]
        else:
            chunksize = int(np.ceil(len(jobs) / self.workers))
            with ProcessPoolExecutor(
                    max_workers=self.workers,
                    initializer=_init, initargs=(shared_arrays,)) as pool:
                result = list(pool.map(function, jobs, chunksize=chunksize))

        _reset_shared_arrays()
        return result

    def starmap(self,
                function: Callable,
                jobs: Sequence[Sequence],
                shared_arrays: SharedArrays = None) -> List:
        """Execution function in parallel for jobs using `starmap`.

        If `self.workers` is 1, the jobs are executed sequentially
        for easier debugging.
        """
        return self.map(partial(_starmaphelper, function), jobs,
                        shared_arrays=shared_arrays)


def _init(shared_arrays: SharedArrays):
    """Set the global objects so that the workers can access shared arrays."""
    global SHARED_ARRAYS  # pylint: disable=global-statement
    SHARED_ARRAYS = shared_arrays


def _reset_shared_arrays():
    """Ensure the shared arrays are reset to None."""
    global SHARED_ARRAYS  # pylint: disable=global-statement
    SHARED_ARRAYS = None


def _starmaphelper(function: Callable, args: Sequence):
    """Assume args is tuple, with first arg being the function."""
    return function(*args)


class Random:
    """Helper class that provides random generators.

    rng is a numpy random generator, pyrng the python random generator.
    """

    def __init__(self, *args, random_state=42, **kwargs):
        self.rng = np.random.default_rng(random_state)
        self.pyrng = random.Random(random_state)
        super().__init__(*args, **kwargs)

    def random_int(self):
        """Shortcut to return a random integer between 0 and 2**32-1."""
        return self.rng.integers(0, 2**32-1)


def last_argmax(array: ArrayLike):
    """Return index of _last_ highest element (instead of first)."""
    arr = np.array(array)
    return len(arr) - arr[::-1].argmax() - 1


def check_datadict(datadict: Union[DataDict, StrictDataDict]) -> StrictDataDict:
    """Check datadict, convert ArrayLike to np.array with at least 1 dim.

    Removes empty keys.
    """
    arraydict = {key: np.atleast_1d(val) for key, val in datadict.items()
                 if val is not None}
    # Second pass to filter empty arrays.
    arraydict = {k: v for k, v in arraydict.items() if len(v) > 0}
    lengths = [len(val) for val in arraydict.values()]
    assert len(set(lengths)) <= 1, f"Lengths must be equal, got `{lengths}`"
    return arraydict


def merge_datadicts(datadicts: Iterable[StrictDataDict]) -> StrictDataDict:
    """Merge data. Can only merge keys that exist in all datadicts."""
    dictlist = list(_d for _d in datadicts if _d)  # Ignore empty dicts.
    if not dictlist:
        return {}
    keys = [set(_d.keys()) for _d in dictlist]
    shared_keys = keys[0].intersection(*keys[1:])
    return {
        key: np.concatenate([_d[key] for _d in dictlist])
        for key in shared_keys
    }


def sample_count(datadict: StrictDataDict) -> int:
    """Return the length of data. Call only on checked datadicts."""
    return len(next(iter(datadict.values()), []))


def sample_diff(data_a: StrictDataDict, data_b: StrictDataDict) -> int:
    """Count number of changes samples between two datadicts.

    Ensure that datadicts are properly formatted or call check_datadict first.

    If len(a) >= len(b), we return the number of samples in a and not in b,
    otherwise the number of samples in b and not in a.

    In other words, return the number of samples replaced plus the change in
    the number of samples (completely new, or completely discarded samples).
    """
    # If one of the data is empty, or if they have different keys, return
    # max possible difference.
    if not (data_a and data_b) or (set(data_a) != set(data_b)):
        return max(sample_count(data_a), sample_count(data_b))
    keys = list(data_a.keys())  # Ensure we use the same order for both.

    # Corner cases solved, now we can iterate efficiently by stacking data.
    diff: Dict[bytes, int] = defaultdict(int)
    for row in _stack_bytes(data_a, keys):
        diff[row] += 1
    for row in _stack_bytes(data_b, keys):
        diff[row] -= 1

    # If len(a) > len(b), a_not_b > b_not_a, because there are more elements.
    # Taking the max implicitly considers the length.
    a_not_b = sum(d for d in diff.values() if d > 0)
    b_not_a = sum(-d for d in diff.values() if d < 0)
    # Note: could easily by extended to return left_diff (dict where d>0),
    # right diff (d<0) and intersection (d=0) instead of summarizing.
    return max(a_not_b, b_not_a)


def _stack_bytes(data: StrictDataDict, keys: Iterable[str]) -> Iterator[bytes]:
    """Stack `data` row bytes in the order determined by `keys`."""
    for index in range(sample_count(data)):
        yield b"".join(data[key][index].tobytes() for key in keys)
