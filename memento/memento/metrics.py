"""Distribution distance metrics.

- jsd: Jensen-Shannon distance between PDFs,
- wsd: Wasserstein p-distance between 1 dimensional CDFs (default p=1),
- wsd_normal: Wasserstein 2-distance for normal distributions.

"""
# pylint: disable=invalid-name

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simps as simpson


def jsd(pdf_1: ArrayLike, pdf_2: ArrayLike, support: ArrayLike = None,
        tolerance: float = 1e-6) -> Union[float, np.ndarray]:
    """Compute the Jensen-Shannon distance (JSD) between pdfs or pmfs.

    pdf_1 and pdf_2 mu_st have the same support, i.e. for all indices i,
    pdf_1[i] and pdf_2[i] mu_st correspond to the same event.
    If support is None, a categorical distribution is assumed, and the pdfs
    are interpreted as pmfs. Resulting probabilties are summed.
    If a support is provided, a continuous distribution is assumed, and
    the resulting probabilities are integrated with respect to the support.

    The JSD is the square root of the Jensen-Shannon Divergence,
    which in turn is related to the Kullback-Leibler Divergence (KL).
    Notable properties:
    - the JSD is symmetrical.
    - the JSD is always finite, even it some probabilities are 0.
    - the JSD is a metric.

    https://en.wikipedia.org/wiki/Jensen–Shannon_divergence

    This functions works with numpy broadcasting, e.g. you can efficiently pass
    only a vector to y1 and a matrix to y2 to compute all distances at once.
    Note that in this case, the rows of the matrix are assumed to be the
    distributions. For mu_ltiple distribution, an array of distances is
    returned, otherwise a scalar.

    If some values are very small, we can get negative number due to limited
    machine precision. We clip these numbers to zero, unless their absolute
    value exceeds tolerance, in which case a RuntimeError is raised.
    """
    pdf_1 = np.atleast_3d(pdf_1)
    pdf_2 = np.atleast_3d(pdf_2).T
    pdf_midpoint = (pdf_1 + pdf_2) / 2
    support = np.array(support) if support is not None else None
    kl_1 = _jsd_kl(pdf_1, pdf_midpoint, support=support, tolerance=tolerance)
    kl_2 = _jsd_kl(pdf_2, pdf_midpoint, support=support, tolerance=tolerance)
    divergence = (kl_1 + kl_2) / 2

    # Take care of machine precision errors, ignore warning during replacement.
    with np.errstate(invalid="ignore"):
        # Negative values should be very small, raise an error otherwise.
        if (divergence < -tolerance).any():
            raise RuntimeError("Negative divergence exceeding tolerance!")
        divergence[divergence < 0] = 0

    distance = np.sqrt(divergence)
    # Return either scalar or vector of results
    try:
        return distance.item()
    except ValueError:
        return distance.squeeze()


def _jsd_kl(pdf: np.ndarray, midpoint: np.ndarray,
            support: Optional[np.ndarray] = None,
            tolerance: float = 1e-6) -> ArrayLike:
    """Kullback-Leibler Divergence with edge cases handling.

    This is not a general KL implementation and should not be used as such.
    It is tailed to work with the JSD above.

    Inputs mu_st be at least 2-D, which is not checked.

    For the limit pdf -> 0, we have pdf log(pdf) -> 0, so we replace these
    values with 0 to avoid numerical issues.

    If the midpoint is 0, both pdfs are 0 at this point, and thus equal.
    Consequently, we define their divergence at this point to be 0.
    This is only for pratical reasons, as it allows us to extend the KL to a
    space where both distributions have no probability, for the KL would be
    undefined otherwise. This makes data handling easier, as we do not need to
    exclude these cases explicitly (which would yield the same result).

    If a support is provided, we use a simpson integral do determine the
    divergence, otherwise a sum.
    """
    pdf_small = np.isclose(pdf, 0, atol=tolerance)
    mid_small = np.isclose(midpoint, 0, atol=tolerance)
    # While we don't use the nan/inf values, we still get warnings -> ignore.
    with np.errstate(invalid="ignore", divide="ignore"):
        kl = np.where(pdf_small | mid_small, 0, pdf * np.log2(pdf / midpoint))
    return (np.sum(kl, axis=1)
            if support is None else simpson(kl, support, axis=1))


def wsd(support: ArrayLike, cdf_1: ArrayLike, cdf_2: ArrayLike,
        p: float = 1.0) -> Union[float, np.ndarray]:
    """Compute the Wasserstein p-distance from 1-dimensional CDFS.

    1-D Wasserstein p-distance = (integral |y1 - y2|^p dx)^(1/p)

    It is important that both y1 and y2 are cdfs, otherwise the result will
    not be correct.

    This functions works with numpy broadcasting, e.g. you can efficiently pass
    only a vector to y1 and a matrix to y2 to compute all distances at once.
    Note that in this case, the rows of the matrix are assumed to be the
    distributions. For mu_ltiple distribution, an array of distances is
    returned, otherwise a scalar.
    """
    cdf_1 = np.atleast_3d(cdf_1)
    cdf_2 = np.atleast_3d(cdf_2).T

    diff = np.abs(cdf_1 - cdf_2)**p
    distances = simpson(diff, support, axis=1)**(1/p)
    try:
        return distances.item()
    except ValueError:
        return distances.squeeze()


def wsd_normal(mu_1: ArrayLike, sigma_1: ArrayLike,
               mu_2: ArrayLike, sigma_2: ArrayLike):
    """Compute the Wasserstein 2-distance for normal distributions."""
    mu_1 = np.atleast_2d(mu_1).T
    mu_2 = np.atleast_2d(mu_2)
    sigma_1 = np.atleast_2d(sigma_1).T
    sigma_2 = np.atleast_2d(sigma_2)
    distances = np.sqrt((mu_1 - mu_2) ** 2 + (sigma_1 - sigma_2) ** 2)
    try:
        return distances.item()
    except ValueError:
        return distances.squeeze()


def minowski(array_1, array_2, p: float = 2.0):
    """Compute the minowski distance between array_1 and array2.
    
    Works with broadcasting to compute pairwise distances between each row
    in array_1 with each row in array_2.
    """
    array_1 = np.atleast_3d(array_1)
    array_2 = np.atleast_3d(array_2).T
    diff = np.abs(array_1 - array_2)**p
    distances = diff.sum(axis=1)**(1/p)
    try:
        return distances.item()
    except ValueError:
        return distances.squeeze()
    