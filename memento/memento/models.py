"""Replay memory classes.

Contains Memento as well as a MultiMemory class that can be used to combine
multiple memories.
"""
# pylint: disable=invalid-name, too-many-ancestors

import logging
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import softmax

from . import bases, utils
from .distances import DistanceKeyError
from .utils import StrictDataDict

logger = logging.getLogger("memory.models")


# Multi Memory.
# =============

class MultiMemory(bases.MemoryBase):
    """Multi-memory class, allowing to combine multiple memories.

    If `get_data` or `get_batches` is called with a not-None `max_samples`,
    you can provide initialize the MultiMemory with `weights`.
    They are normalized to `1` and determine which share of `max_samples` is
    requested from each memory.
    If the `get_*` methods are called without `max_samples`, this has no effect.
    """
    __initialized: bool = False  # Protect against recursion during init.

    def __init__(self,
                 memories: Iterable[bases.MemoryBase] = None,
                 weights: Optional[Iterable[float]] = None,
                 ):  # pylint: disable=super-init-not-called
        if memories is None:
            raise TypeError("You must provide memories.")
        self.memories = list(memories)
        if weights is not None:
            _weights = list(weights)
        else:
            _weights = [1] * len(self.memories)
        _total = sum(_weights)
        self.weights = [_weight / _total for _weight in _weights]
        self.__initialized = True

    @property
    def _memiter(self):
        """Iterate over memories, adding log output."""
        for mem in self.memories:
            logger.debug("Memory: `%s`.", mem)
            yield mem

    def select(self, *args, **kwargs):
        """Does not select samples on its own."""
        raise RuntimeError("MultiMemory does not select batches itself.")

    # Data interfaces.
    def insert(self, *datadicts, **samples):
        """Provide all samples to all memories."""
        logger.debug("Adding samples to all memories.")
        for mem in self._memiter:
            mem.insert(*datadicts, **samples)

    def get(self, max_samples=None):
        """Let memories choose samples, adjust max_samples to weight."""
        logger.debug("Calling `get_batches` of all memories.")
        results = []
        for mem, _max in zip(self._memiter, self._mem_samples(max_samples)):
            results.append(mem.get(max_samples=_max))
        return utils.merge_datadicts(results)

    def _mem_samples(self, max_samples: Optional[int]) -> List[Optional[int]]:
        """Get max samples per memory."""
        if max_samples is None:
            return [None] * len(self.memories)
        return [int(weight * max_samples) for weight in self.weights]

    # Helpers to work with the predictors.

    def predict(self, *args, **kwargs):
        """Cannot naively merge predictions."""
        raise RuntimeError("MultiMemory cannot predict itself.")

    def update_predictor(self, predictor):
        logger.debug("Updating memory predictors.")
        for mem in self._memiter:
            mem.update_predictor(predictor)

    # Helpers to delegate all other attributes to memories.

    def __getattr__(self, name: str) -> Tuple:
        """Delegate to all sub-memories."""
        return tuple(getattr(mem, name) for mem in self.memories)

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets attribute for all sub memories, except for `data`.

        Special case: when setting `data`, expect a tuple, split it up and
        set each sub-memory with the corresponding samples.
        This allows easily saving/loading batches for a multimem.
        """
        if self.__initialized:  # Prevent recursion during __init__.
            if name == "data":
                for mem, mem_data in zip(self.memories, value):
                    setattr(mem, name, mem_data)
            else:
                for mem in self.memories:
                    setattr(mem, name, value)
        else:
            object.__setattr__(self, name, value)

    # Except methods needed to pickle.

    def __getstate__(self):
        return (self.memories, self.weights, self.__initialized)

    def __setstate__(self, state):
        self.memories, self.weights, self.__initialized = state


# Replay Memory.
# ==============
# Distance functions map a sequence of data batches to a distance matrix.

DISTANCETYPE = Callable[[Sequence[StrictDataDict],
                         Optional[Sequence[StrictDataDict]]], ArrayLike]
AGGTYPE = Callable[[ArrayLike], np.ndarray]


class Memento(utils.Random, bases.MemoryBase):
    """The Memento memory based on coverage maximization.

    - Batches data.
    - Computes pairwise distances between batches.
    - Uses KDE to estimate density from distances.
    - Selects samples that approximate uniform density, i.e. maximize coverage.

    By default, samples are replaced randomly with a probability to be replaced
    of softmax(density / temperature). A higher density leads to a higher
    probability to be replaced. A higher temperature smoothens the distribution,
    while a lower temperature assigns more probability to the highest-density
    batch. If the temperature is set to 0, Memento deterministically removes
    the highest-density batch.

    Distances are pluggable to adapt to different output and prediction types,
    e.g. categorical/continual variables and point-estimates or distribution
    estimates.
    """

    def __init__(self, *args,
                 bw: float = 0.1,
                 temperature: float = 0.01,
                 forget: float = 0.0,
                 distances: Sequence[DISTANCETYPE] = None,
                 require_all_distances: bool = False,
                 # For Memento, batching is not optional.
                 batching_size: int = 256,
                 insert_chunksize=None,
                 **kwargs):
        """Initialize kernel bandwidth."""
        self.bw = bw
        self.temperature = temperature
        self.distances = distances if distances is not None else []
        self.require_all_distances = require_all_distances

        # Experimental features.
        self.insert_chunksize = insert_chunksize
        self.forget = forget
        # End of experimental.
        super().__init__(*args, batching_size=batching_size, **kwargs)

    def insert_datadict(self, datadict):
        """If datadict is larger than insert_chunksize, split it up.

        We need to compute pairwise distances which takes quadradtic memory
        space. This way, we can bound the memory requirements.
        """
        if self.insert_chunksize is None:
            return super().insert_datadict(datadict)
        total = utils.sample_count(datadict)
        logger.debug("Inserting at most `%i` sample at once.",
                     self.insert_chunksize)
        logger.debug("Inserting `%i` chunks of samples.",
                     np.ceil(total / self.insert_chunksize))
        _last_forget = self.forget
        for i, start in enumerate(range(0, total, self.insert_chunksize)):
            super().insert_datadict({k: v[start:start+self.insert_chunksize]
                                     for k, v in datadict.items()})
            if i > 0:
                self.forget = 0.0  # Don't forget twice.
        self.forget = _last_forget

    def select(self, max_samples, current_data, new_data):
        """Select bactches to maximize space coverage."""
        total_current = utils.sample_count(current_data)
        total_new = utils.sample_count(new_data)
        if current_data and (self.forget > 0):
            # Experimental feature.
            # Do not forget more batches than incoming data.
            naive_forget = int(np.ceil(total_current * self.forget))
            num_forget = min(naive_forget, total_new)
            logger.debug("Forgetting `%i` samples.", num_forget)
            keep_index = self.rng.choice(
                total_current, total_current - num_forget, replace=False)
            current_data = {k: v[keep_index] for k, v in current_data.items()}

        logger.debug("Batching samples.")
        batches = self.batch(utils.merge_datadicts([
            current_data, new_data
        ]))

        logger.debug("Computing distance matrices and applying kernel.")
        # Ensure all distance matrices are up-to-date.
        # (Rely on the distances to cache old results for performance)
        matrices = self.compute_matrices(batches)
        if not matrices:  # Fall back to FIFO if there are no distances.
            return {k: v[:max_samples]
                    for k, v in utils.merge_datadicts(batches).items()}

        logger.debug("Computing densities.")
        # We store the sum, which is easier to update, and compute mean later.
        densities = [matrix.sum(axis=1) for matrix in matrices]

        logger.debug("Removing batches.")
        if self.temperature == 0:
            logger.debug("Selecting highest density batch deterministically.")
        else:
            logger.debug("Selecting randomly with temperature `%e`.",
                         self.temperature)

        # Use a mask to "remove" elements without reallocating memory.
        mask = np.ones(len(batches), dtype=bool)  # All true first.
        indices = np.arange(len(batches))
        current_samples = sum(utils.sample_count(_b) for _b in batches)
        while current_samples > max_samples:
            assert len(batches) == mask.sum()
            # Now divide by number of samples to get mean from sum.
            n_batches = len(batches)
            combined = self.agg([dens[mask] / n_batches for dens in densities])

            if self.temperature == 0:
                # Highest local density (last/newest batch if equal).
                to_remove = utils.last_argmax(combined)
            else:
                # Use softmax to determine probabilities for replacement.
                # High density -- high probability
                probs = softmax(combined / self.temperature)
                to_remove = self.rng.choice(len(probs), p=probs)

            # Remove and mask batch (careful: using the original index).
            original_index = indices[mask][to_remove]

            current_samples -= utils.sample_count(batches[to_remove])
            del batches[to_remove]
            mask[original_index] = False
            # Update remaining densities inplace.
            for density, matrix in zip(densities, matrices):
                density[mask] -= matrix[mask, original_index]

        return utils.merge_datadicts(batches)

    def coverage_change(self, reference_data, current_data=None, predict=True,
                        return_stats=False, return_densities=False):
        """Compute how much coverage has increased.

        If current_data is None, use data current in memory.
        Returns a relative value. A result of 0 means that reference and
        current data cover the same space; a result of 1 means that all space
        covered by current data is "new",  i.e. not covered by the reference.
        """
        if not self.distances:
            raise RuntimeError(
                "Cannot compute coverage change without distances.")

        if predict:
            reference_data = self.add_predictions(reference_data)
            if current_data is not None:
                current_data = self.add_predictions(current_data)

        ref_batches = self.batch(reference_data)
        cur_batches = self.batch(current_data
                                 if current_data is not None else self.data)

        if not ref_batches:
            raise RuntimeError(
                "Not enough reference data to compute coverage change.")
        if not cur_batches:
            raise RuntimeError(
                "Not enough current data to compute coverage change.")

        # Compute density of current samples w.r.t reference (the update).
        ref_density = self.agg([
            mat.mean(axis=1)
            for mat in self.compute_matrices(cur_batches, ref_batches)
        ])
        # And w.r.t to itself (the baseline).
        cur_density = self.agg([mat.mean(axis=1) for mat in
                                self.compute_matrices(cur_batches)])

        # Approximate the area/volume of sample space that has changed.
        # Approximate increased area by summing all density increases.
        # (Only increases, as otherwise we count every change twice.)
        change = cur_density - ref_density
        total_increase = np.sum(change[change > 0])
        # Approximate the total area of current sample space.
        total_coverage = np.sum(cur_density)
        relative_increase = total_increase / total_coverage

        if not return_stats:
            return relative_increase

        # Extended statistics.
        stats = {
            'coverage_increase': relative_increase,
            'total_coverage_increase': total_increase,
            'total_coverage': total_coverage,
        }
        if return_densities:
            # Also return the sorted densities to analyse memory contents.
            # Implementation detail: list(tuple) for compatibility with pandas and csv storage.
            stats['densities'] = [tuple(np.sort(cur_density)[::-1])]

        return relative_increase, stats

    def kernel(self, x: ArrayLike):
        """Apply a gaussian kernel elementwise to x.

        The kernel is one dimensional and has std 1. Scaling is done via
        bandwidth, not kernel standard deviation.
        """
        x = np.array(x) / self.bw
        return np.exp(- 0.5 * (x ** 2)) / np.sqrt(2 * np.pi)

    def agg(self, vectors: ArrayLike):
        """Aggregate multiple density vectors into one.

        We only want to consider samples that are in high density area w.r.t
        *all* distances, thus we select the *minimum* density.
        """
        return np.min(vectors, axis=0)

    def compute_matrices(
            self, batches: Sequence[StrictDataDict],
            others: Optional[Sequence[StrictDataDict]] = None) \
            -> Sequence[np.ndarray]:
        """Compute all distance matrices and apply kernel."""
        matrices = []
        for distance in self.distances:
            try:
                logger.debug("Next distance: `%s`.", str(distance))
                matrix = np.array(distance(batches, others))
                matrices.append(self.kernel(matrix))
            except DistanceKeyError as error:
                if self.require_all_distances:
                    raise error
                # Otherwise, skip unavailable distances.
        return matrices
