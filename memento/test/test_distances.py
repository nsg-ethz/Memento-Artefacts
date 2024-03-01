"""Test distance computation."""
# pylint: disable=redefined-outer-name
# (because of fixtures)

import numpy as np
import pytest
from scipy import integrate, stats

from memento import distances, metrics


# Test base classes.
# ==================

class _DistributionDistance(distances.DistributionDistance):
    def compute_distribution(self, values):
        """Turn values into 1d arrays (whether 0, 1, or 2d.)."""
        out = np.atleast_2d(values).mean(axis=0)
        assert out.ndim == 1
        return out

    def distribution_distance(self, dist, other_dists):
        """Return absolute distance."""
        return [np.sum(np.abs(dist - other)) for other in other_dists]


@pytest.fixture
def distancebase():
    """Return a distribution distance base class."""
    return _DistributionDistance


@pytest.mark.parametrize('jobs', [None, 3])
@pytest.mark.parametrize('attr', ['x', 'y', 'yhat'])
def test_pairwise_distances(distancebase, attr, jobs):
    """Test the computation of the pairwise distance matrix."""
    batches = [{attr: [val]} for val in [10, 100, 1000]]
    distance_computer = distancebase(var=attr, workers=jobs)
    result = distance_computer(batches)
    np.testing.assert_array_equal(result, [
        [0, 90, 990],
        [90, 0, 900],
        [990, 900, 0],
    ])


@pytest.mark.parametrize('jobs', [None, 3])
@pytest.mark.parametrize('attr', ['x', 'y', 'yhat'])
def test_distances(distancebase, attr, jobs):
    """Test the computation of an arbitrary distance matrix."""
    batches = [{attr: [val]} for val in [10, 100, 1000]]
    others = [{attr: [val]} for val in [5, 50]]

    distance_computer = distancebase(var=attr, workers=jobs)
    result = distance_computer(batches, others)
    # Expected: axis 0: batches, axis 1: others, value: absolute diff.
    np.testing.assert_array_equal(result, [
        [5, 40],
        [95, 50],
        [995, 950],
    ])


@pytest.mark.parametrize("attrbatch, attrdist", [
    ("y", "x"),
    ("x", "y"),
    ("x", "yhat"),
    ("x", "lalala"),
])
def test_error(distancebase, attrbatch, attrdist):
    """Test custom errors."""
    batches = [{attrbatch: [val]} for val in [10, 100, 1000]]
    distance_computer = distancebase(var=attrdist)
    with pytest.raises(distances.DistanceKeyError):
        distance_computer(batches)
    with pytest.raises(KeyError):  # It's still a KeyError.
        distance_computer(batches)


# Test implementations.
# =====================

@pytest.mark.parametrize("jsdclass", [
    distances.JSDCategoricalPoint, distances.JSDCategoricalDistribution
])
def test_jsd_categorical(jsdclass, mocker):
    """Test initialization and call of jsd mixin."""
    model = jsdclass('x', jsd_tolerance=42, classes=[1, 2])
    jsd = mocker.patch(metrics.__name__ + ".jsd")
    model.distribution_distance(1, [2])
    jsd.assert_called_with(1, [2], tolerance=42, support=None)


@pytest.mark.parametrize("jsdclass", [
    distances.JSDContinuousPoint, distances.JSDContinuousDistribution
])
def test_jsd_interval(jsdclass, mocker):
    """Test call of jsd mixin with support."""
    support = [21, 42]
    model = jsdclass('x', support=support, jsd_tolerance=42)
    jsd = mocker.patch(metrics.__name__ + ".jsd")
    model.distribution_distance(1, [2])
    # We cannot check the array in the call easily, so check individually.
    call = jsd.mock_calls[0]
    assert call.args == (1, [2])
    assert call.kwargs['tolerance'] == 42
    assert list(call.kwargs['support']) == support


def test_label_distribution():
    """Test that label distributions are computed correctly."""
    model = distances.JSDCategoricalPoint('y', classes=3)
    np.testing.assert_allclose(model.compute_distribution([0, 2, 1, 2]),
                               [1/4, 1/4, 2/4])
    model = distances.JSDCategoricalPoint('y', classes=4)
    np.testing.assert_allclose(model.compute_distribution([0, 2, 1, 2]),
                               [1/4, 1/4, 2/4, 0])


def test_label_distribution_mixture():
    """Test that label distributions are computed correctly."""
    model = distances.JSDCategoricalDistribution('y', classes=2)
    np.testing.assert_allclose(
        model.compute_distribution([[1, 0], [0, 1]]),
        [1/2, 1/2]
    )
    model = distances.JSDCategoricalDistribution('y')
    np.testing.assert_allclose(
        model.compute_distribution([[1, 0, 0], [1, 0, 0], [0, 0, 1]]),
        [2/3, 0, 1/3]
    )


def test_label_distribution_mixture_wrong_size():
    """Test that label distributions are computed correctly."""
    with pytest.raises(AssertionError):
        model = distances.JSDCategoricalDistribution('y', classes=3)
        np.testing.assert_allclose(
            model.compute_distribution([[1, 0], [0, 1]]), [1/2, 1/2])


def test_interval_distribution_kde():
    """Test KDE estimate of interval distribution."""
    support = np.linspace(-1.1, 1.1, 100)
    dist = stats.truncnorm(-1, 1)
    values = dist.rvs(size=10000, random_state=42)
    expected = dist.pdf(support)

    model = distances.JSDContinuousPoint('y', support=support)
    estimate = model.compute_distribution(values)

    # Test that result integrates to 1.
    assert integrate.simps(estimate, support) == pytest.approx(1, 10**-3)

    # Also check integrated error relative to expected result is less than 10%
    assert integrate.simps(np.abs(estimate - expected), support) <= 0.1


def test_interval_distribution_kde_clip():
    """Test KDE estimate clips values outside support."""
    _a, _b = -2, 2
    support = np.linspace(_a, _b, 100)
    dist = stats.norm()
    values = dist.rvs(size=10000, random_state=42)
    assert any(values <= _a) or any(values >= _b)  # Something to clip.

    expected = dist.pdf(support)
    # Effect from clipping: Everything above/below is at the edge now.
    expected[0] += dist.cdf(_a)
    expected[-1] += dist.sf(_b)

    model = distances.JSDContinuousPoint('y', support=support)
    estimate = model.compute_distribution(values)

    # Test that result integrates to 1.
    assert integrate.simps(estimate, support) == pytest.approx(1, 10**-3)

    # Also check integrated error relative to expected result is less than 10%
    assert integrate.simps(np.abs(estimate - expected), support) <= 0.1


def test_interval_distribution_mixture():
    """Test that label distributions are computed correctly."""
    model = distances.JSDContinuousDistribution('y', support=[0, 1])
    np.testing.assert_allclose(
        model.compute_distribution([[1, 0], [0, 1]]),
        [1/2, 1/2]
    )
    model = distances.JSDContinuousDistribution('y', support=[0, 1, 2])
    np.testing.assert_allclose(
        model.compute_distribution([[1, 0, 0], [1, 0, 0], [0, 0, 1]]),
        [2/3, 0, 1/3])


def test_minowski():
    """Test the minowski distance class."""
    model = distances.Minowski('x', p=2)
    batches = [
        {'x': [[1, 2, 3]]},  # one row, take as-is
        {'x': [  # Two rows, take average ([1, 5, 7])
            [1, 4, 6],
            [1, 6, 8],  
        ]}
    ]
    np.testing.assert_allclose(
        model(batches),
        [[0, 5], [5, 0]]
    )
