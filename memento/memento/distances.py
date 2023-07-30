"""Classes to compute pairwise distribution distances between data batches."""
# pylint: disable=invalid-name
# A few names like bw and capital LABELS annoy pylint.

import abc
import ctypes
import logging
import multiprocessing as mp
import warnings
from functools import partial
from typing import Generic, Iterable, Optional, Sequence, Tuple, TypeVar, Union

import KDEpy
import numpy as np
from scipy import integrate

from . import metrics, utils
from .utils import ArrayLike, StrictDataDict

logger = logging.getLogger("memory.distances")


class DistanceKeyError(KeyError):
    """Custom error class to allow to pinpoint missing datadict keys."""


# Distance computer bases.
# ========================
DIST = TypeVar("DIST", bound=ArrayLike)


class DistributionDistance(utils.MultiProcessing, abc.ABC,
                           Generic[DIST]):
    """Base class to for batch distribution distances.

    Distances are computed for a single feature with a 1-dimensional
    distribution. Batches are assumed to be dicts of numpy arrays. The `var`
    argument to this class determines for which key the distance is computed.

    Subclasses need to implement:
    - `compute_distribution` to compute a distribution from an array of values.
    - `distribution_distance` to compute the distance between distributions.
    """

    def __init__(self, var: str, *args, **kwargs):
        self.var = var  # Which DataDict variable to fetch.
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}({self.var})"

    @abc.abstractmethod
    def compute_distribution(self, values: np.ndarray) -> DIST:
        """Return a one-dimensional distribution for the given values."""

    @abc.abstractmethod
    def distribution_distance(self, dist: DIST, other_dists: Sequence[DIST]) \
            -> ArrayLike:
        """Compute the distances between dist and each other dists."""

    def __call__(self,
                 batches: Sequence[StrictDataDict],
                 others: Optional[Sequence[StrictDataDict]] = None) \
            -> np.ndarray:
        """Compute the pairwise distances."""
        if others is None:
            others = []
        logger.debug("Checking inputs.")
        checked = [utils.check_datadict(_b) for _b in batches]
        checked_others = [utils.check_datadict(_b) for _b in others]

        logger.debug("Fetching values.")
        values = self.get_values(checked)
        other_values = self.get_values(checked_others)

        logger.debug("Computing distributions.")
        distributions = self.compute_distributions(values)
        other_distributions = self.compute_distributions(other_values)

        if not others:
            logger.debug("Computing pairwise distances.")
            return self.compute_pairwise_distances(distributions)
        else:
            logger.debug("Computing distances.")
            return self.compute_distances(distributions, other_distributions)

    def get_value(self, batch: StrictDataDict) -> np.ndarray:
        """Get self.var, raise DistanceKeyError if it is missing."""
        try:
            return batch[self.var]
        except KeyError as err:
            # Provide more informative logs for known variables.
            if self.var == 'x':
                message = "Batch does not contain inputs (`x`)."
            elif self.var == 'y':
                message = "Batch does not contain outputs (`y`)."
            elif self.var == 'yhat':
                message = "Batch does not contain predictions (`yhat`)."
            else:
                message = f"Batch does not contain variable `{self.var}`."
            logger.critical(message)
            raise DistanceKeyError(message) from err

    def get_values(self, batches: Sequence[StrictDataDict]) \
            -> Sequence[np.ndarray]:
        """Get value from batches."""
        return [self.get_value(batch) for batch in batches]

    def compute_distributions(self, values: Sequence[np.ndarray]) \
            -> Sequence[DIST]:
        """Get 1-dimensional distributions from values."""
        return [self.compute_distribution(val) for val in values]

    def compute_pairwise_distances(self, dists: Sequence[DIST]) -> np.ndarray:
        """Compute a matrix of pairwise distances.

        As the distance matrix is symmetric, only the lower triangle is computed
        and subsequently mirrored.

        Each column of the lower triangle is computed in a separate process, if
        `workers` is not None.
        """
        total = len(dists)
        dist_arrays = [np.array(dist) for dist in dists]
        dist_size = len(dist_arrays[0])

        # Create shared arrays and load distributions.
        shared_input = mp.RawArray(ctypes.c_double, total * dist_size)
        shared_output = mp.RawArray(ctypes.c_double, total * total)
        _dists = _load_shared_array(shared_input, (total, dist_size))
        for ind, distribution in enumerate(dist_arrays):
            _dists[ind, :] = distribution

        # Prepare jobs to compute distances.
        function = partial(_pairwise_distances,
                           function=self.distribution_distance,
                           total=total,
                           dist_size=dist_size)
        shared = [shared_input, shared_output]
        self.map(function, list(range(total)), shared_arrays=shared)
        return _load_shared_array(shared_output, (total, total))

    def compute_distances(self,
                          src_dists: Sequence[DIST],
                          tgt_dists: Sequence[DIST]):
        """Compute distance between different sets of 1-d distributions.

        The output matrix has shape (len(src_dists), len(tgt_dists)).
        """
        total_src = len(src_dists)
        total_tgt = len(tgt_dists)
        src_arrays = [np.array(dist) for dist in src_dists]
        tgt_arrays = [np.array(dist) for dist in tgt_dists]
        dist_size = len(src_arrays[0])

        # Create shared arrays and load distributions.
        shared_src = mp.RawArray(ctypes.c_double, total_src * dist_size)
        shared_tgt = mp.RawArray(ctypes.c_double, total_tgt * dist_size)
        shared_output = mp.RawArray(ctypes.c_double, total_src * total_tgt)

        _src = _load_shared_array(shared_src, (total_src, dist_size))
        for ind, distribution in enumerate(src_arrays):
            _src[ind, :] = distribution

        _tgt = _load_shared_array(shared_tgt, (total_tgt, dist_size))
        for ind, distribution in enumerate(tgt_arrays):
            _tgt[ind, :] = distribution

        # Prepare jobs to compute distances.
        function = partial(_distances,
                           function=self.distribution_distance,
                           total_src=total_src, total_tgt=total_tgt,
                           dist_size=dist_size)
        shared = [shared_src, shared_tgt, shared_output]
        self.map(function, list(range(total_src)), shared_arrays=shared)
        return _load_shared_array(shared_output, (total_src, total_tgt))


# Distance computer implementations.
# ==================================

class JSDCategoricalPoint(DistributionDistance):
    """Compute distributions and distances between categorical values/labels.

    Batch values are assumed to be integer labels, and you need to provide
    either the number of classes or a list of all possible labels.

    The computation works in two steps.
    1. Compute empirical distributions for the batch values.
    2. Compute the JSD distances between the empirical distributions.
    """
    REQUIRE_LABELS = True
    LABELS: Sequence = []  # Can be used for static assignment of classes.

    def __init__(self, *args, classes: Union[int, Iterable] = None,
                 jsd_tolerance: float = 1e-6, **kwargs):
        self.__tol = jsd_tolerance
        if classes is not None:
            if isinstance(classes, int):
                self.LABELS = list(range(classes))
            else:
                self.LABELS = list(classes)  # Iterate only once.
        elif self.REQUIRE_LABELS and not self.LABELS:
            raise ValueError("No class labels.")
        super().__init__(*args, **kwargs)

    def compute_distribution(self, values: ArrayLike) -> ArrayLike:
        """Compute the empirical pmf for all possible labels."""
        label_counts = dict(zip(*np.unique(values, return_counts=True)))
        all_counts = [label_counts.get(label, 0) for label in self.LABELS]
        return np.array(all_counts) / np.sum(all_counts)

    def distribution_distance(
        self, dist: ArrayLike, other_dists: Sequence[ArrayLike]
    ) -> ArrayLike:
        """Compute the distances between dist and each other dist."""
        return metrics.jsd(
            dist, np.array(other_dists), tolerance=self.__tol, support=None)


class JSDCategoricalDistribution(JSDCategoricalPoint):
    """Compute distances between categorical distributions.

    Batch values are assumed to be distributions over a fixed set of classes.
    You can provide the number of classes to __init__. If you do, the batch
    distributions are checked to ensure they have the correct size.

    The computation works in two steps.
    1. Compute a mixture distribution for each batch. Each invidual value
       in each batch is expected to be a distribution. The mixture combines
       these distributions with equal weights, i.e. it is the average.
    2. Compute the JSD distances between the mixture distributions.
    """
    REQUIRE_LABELS = False  # Optional as we already expect distributions.

    def compute_distribution(self, values: ArrayLike) -> ArrayLike:
        """Return a one-dimensional distribution for the batch.

        Concretely, compute a mixture distribution by averaging over axis 0,
        resulting in a single mixture distribution per batch.
        If explicit labels are provided, ensure correct size.
        """
        mean = np.mean(values, axis=0)
        if self.LABELS:
            assert len(mean) == len(self.LABELS)
        return mean


class _JSDContinousBase(DistributionDistance):
    """Compute distribution between continuous distributions.

    The pdfs must be provided on a fixed support that is identical for all
    distributions. For example, if your support is [0, 1, 4], pdfs are assumed
    to be arrays containing [pdf(0), pdf(1), pdf(4)]. The support values are
    not required to be equidistant, but they must be sorted.
    """

    def __init__(self, *args,
                 support: ArrayLike,  # Support is mandatory keyword argument.
                 jsd_tolerance: float = 1e-6, **kwargs):
        self.__tol = jsd_tolerance
        self.support = np.array(support)
        super().__init__(*args, **kwargs)

    def distribution_distance(
        self, dist: ArrayLike, other_dists: Sequence[ArrayLike]
    ) -> ArrayLike:
        """Compute the distances between dist and each other dist."""
        return metrics.jsd(dist, np.array(other_dists),
                           tolerance=self.__tol, support=self.support)


class JSDContinuousPoint(_JSDContinousBase):
    """Compute distribution distances between continuous values.

    For each batch, the pdf is estimated via Kernel Density Estimation (KDE).

    The support may be provided in two ways:
    1. As an array of support values. This is the default.
    2. As a tuple (n_points, mass). In this case, the support is created
        as an equidistant grid of n_points such that a mass fraction of
        values lies on the support (mass must be between 0 and 1).


    By default, values outside of the provided support are clipped to the
    closest value, as the underlying KDE library (KDEpy) cannot handle values
    outside of the support. Ensure that your support covers as much of the
    distribution as possible to get the most accurate results.

    Some of the more advanced bandwidth estimators of KDEpy may fail, e.g. if
    the batch does not contain sufficiently diverse values. By default,
    we fall back to a simple bandwidth estimate (a tenth of the support range)
    to avoid crashes.
    Finally, you may specify `global_bw`. If True, only a single bandwidth is
    estimated across all batches. By default bandwidths are estimated
    individually for each batch.
    """

    def __init__(self, *args,
                 support: Union[ArrayLike, Tuple[int, float]],
                 clip_to_support=True, bw='silverman', kde_kwargs=None,
                 global_bw=False, fallback_bw=True, clip_eps=1e-6, **kwargs):
        if isinstance(support, tuple):
            self.auto_support = support
            support = []
        else:
            self.auto_support = None

        super().__init__(*args, support=support, **kwargs)

        self.clip_to_support = clip_to_support
        self.clip_eps = clip_eps

        self.kde_kwargs = {} if (kde_kwargs is None) else kde_kwargs

        if global_bw:
            self.global_bw = bw
            self.bw = None
        else:
            self.global_bw = None
            self.bw = bw
        self.fallback_bw = fallback_bw

    def compute_distribution(self, values: ArrayLike) -> ArrayLike:
        """Compute output pdf using KDE.

        If `self.clip_to_support` is True, clip the array to the min/max of the
        provided support. This only affects values below the min/above the max.
        Note that they are clipped not exactly to the support, but with some
        small epsilon in between as KDEpy does not support data points on the
        grid edges [1].

        Importantly, the clipping is a feature only intended to ensure that the
        KDE can be computed on the support, even if there are _rare_ outliers.
        The support should cover the whole distribution. If any significant
        part of the sample lies outside the support, the clipping will likely
        result in undesirable distribution estimates.

        [1]: https://github.com/tommyod/KDEpy/issues/7
        """
        if self.clip_to_support:
            logger.debug("Clipping values to support.")
            values = np.clip(values,
                             self.support[0] + self.clip_eps,
                             self.support[-1] - self.clip_eps)

        # KDEpy can struggle if values have a large range, so we scale them
        # to 0-1. This is not a problem as the support is also scaled.
        shift = self.support[0]
        scale = self.support[-1] - self.support[0]
        _x = (self.support - shift) / scale
        values = (values - shift) / scale

        with warnings.catch_warnings(), np.errstate(divide='ignore'):
            warnings.simplefilter("ignore")
            logger.debug("Fitting with bandwidth `%s`.", self.bw)
            try:
                pdf = KDEpy.FFTKDE(
                    bw=self.bw, **self.kde_kwargs).fit(values).evaluate(_x)
            except Exception as error:  # pylint: disable=broad-exception-caught
                if not self.fallback_bw:
                    raise error
                fallback_bw = 0.1
                logger.debug("Fitting failed, falling back to bandwidth `%s`.",
                             fallback_bw)
                pdf = KDEpy.FFTKDE(
                    bw=fallback_bw, **self.kde_kwargs).fit(values).evaluate(_x)

        # Ensure pdf integrates to 1.
        return pdf / integrate.simps(pdf, self.support)

    def compute_distributions(self, values: Sequence[ArrayLike]):
        """Parallelize KDE."""
        if self.auto_support is not None:
            # Re-compute support for provided values.
            n_points, mass = self.support
            assert 0 < mass <= 1
            residual = (1 - mass) / 2
            low, high = np.quantile(np.array(values), [residual, 1-residual])
            self.support = np.linspace(low, high, int(n_points))

        if self.global_bw is not None:
            # Determine fixed bandwidth across all batches.
            self.bw = KDEpy.FFTKDE(bw=self.global_bw).fit(values).bw

        return self.map(self.compute_distribution, values)


class JSDContinuousDistribution(_JSDContinousBase):
    """Compute distances between continuous distributions."""

    def compute_distribution(self, values: ArrayLike) -> ArrayLike:
        """Return a one-dimensional distribution for the batch.

        Concretely, compute a mixture distribution by averaging over axis 0.
        For more details, see JSDCategoricalDistribution.
        """
        return np.mean(values, axis=0)


# Helpers for multiprocessing with shared arrays.
# ===============================================

def _pairwise_distances(index, function=None,
                        total=None, dist_size=None):
    """Computer distances from shared input and write to shared output."""
    distributions = _load_shared_array(
        utils.SHARED_ARRAYS[0], (total, dist_size))
    distances = _load_shared_array(
        utils.SHARED_ARRAYS[1], (total, total))

    row_map = np.arange(total)
    row = row_map[index]
    other_rows = row_map[index:]

    # Assign column and row at once (for pairwise distances, they are identical)
    distances[other_rows, row] = distances[row, other_rows] = function(
        distributions[row], distributions[other_rows])


def _distances(index, function=None,
               total_src=None, total_tgt=None, dist_size=None):
    """Compute distances between shared src and tgt, write to shared output."""
    src_distributions = _load_shared_array(
        utils.SHARED_ARRAYS[0], (total_src, dist_size))
    tgt_distributions = _load_shared_array(
        utils.SHARED_ARRAYS[1], (total_tgt, dist_size))
    distances = _load_shared_array(
        utils.SHARED_ARRAYS[2], (total_src, total_tgt))

    distances[index, :] = function(src_distributions[index], tgt_distributions)


def _load_shared_array(shared_array, shape):
    return np.frombuffer(shared_array).reshape(shape)
