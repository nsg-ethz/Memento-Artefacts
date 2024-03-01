"""Test distance metrics."""

import numpy as np
import pytest

from memento import metrics


@pytest.mark.parametrize("distribution1,distribution2,distance", [
    ([1, 0, 0], [1, 0, 0], 0.),
    ([1, 0, 0], [0, 1, 0], 1.),
    ([1, 0, 0], [0, 0, 1], 1.),
])
def test_jsd(distribution1, distribution2, distance):
    """Compute simple metrics."""
    jsd = metrics.jsd(distribution1, distribution2)
    assert jsd == pytest.approx(distance)


def test_jsd_zero_prob():
    """For convenience, we the same elements in both vectors to be zero.

    This should not result in Infs or NaNs.
    """
    assert metrics.jsd([1, 0], [1, 0]) == 0
    assert metrics.jsd([1, 0, 0], [0, 0, 1]) == 1


def test_jsd_broadcasting():
    """We can compute many metrics at once using broadcasting."""
    metrics_1 = metrics.jsd([1, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(metrics_1, [0, 1, 1])
    metrics_2 = metrics.jsd([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 0, 0])
    assert np.allclose(metrics_2, [0, 1, 1])
    all_metrics = metrics.jsd([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(all_metrics, [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])


def test_wsd():
    """Test wasserstein distance for cdfs."""
    # We use long vectors to approximate a delta function for which we can tell
    # the distance.
    n = 100000
    support = np.linspace(0, 1, n)
    y1 = np.ones_like(support)
    y2 = np.zeros_like(support)
    y2[-1] = 1

    assert metrics.wsd(support, y1, y1) == pytest.approx(0)
    assert metrics.wsd(support, y1, y2) == pytest.approx(1, rel=1e-5)


def test_wsd_broadcasting():
    """Test that broadcasting works."""
    support = [0, 1, 2]
    data1 = np.random.random_sample((4, 3))
    data2 = np.random.random_sample((5, 3))

    result = metrics.wsd(support, data1, data2)
    assert result.shape == (4, 5)
    assert np.allclose(result, metrics.wsd(support, data2, data1).T)

    for i in range(4):
        for j in range(5):
            assert result[i, j] == metrics.wsd(support,
                                               data1[i, :], data2[j, :])

    # Test 1d as well.
    assert metrics.wsd(support, data1[0, :], data2).shape == (5,)
    assert metrics.wsd(support, data1, data2[0, :]).shape == (4,)


@pytest.mark.parametrize("mu1, sigma1, mu2, sigma2, distance", [
    (1, 0, 0, 0, 1),
    (1, 1, 0, 1, 1),
    (1, 1, 0, 2, np.sqrt(2)),
    (4, 7, 1, 3, 5)
])
def test_wsd_normal(mu1, sigma1, mu2, sigma2, distance):
    """Test computation of the wasserstein distance for normal dists."""
    assert metrics.wsd_normal(mu1, sigma1, mu2, sigma2) == \
        pytest.approx(distance)


def test_wsd_normal_broadcasting():
    """Test broadcasting."""
    mu1 = [1, 2]
    mu2 = [3, 4]
    sigma1 = [5, 6]
    sigma2 = [7, 8]

    assert np.allclose(metrics.wsd_normal(0, 0, mu1, sigma1),
                       metrics.wsd_normal(mu1, sigma1, 0, 0))

    result = metrics.wsd_normal(mu1, sigma1, mu2, sigma2)
    for i in range(2):
        for j in range(2):
            assert result[i, j] == metrics.wsd_normal(mu1[i], sigma1[i],
                                                      mu2[j], sigma2[j])


@pytest.mark.parametrize("array_1, array_2, p, expected", [
    [[1, 2], [1.1, 2.2], 1, 0.3],
    [[1, 1, 3], [1, 4, 7], 2, 5],
    [[[1, 2], [4, 6]], [1, 2], 2, [0, 5]],  # broascasting
])
def test_minowski(array_1, array_2, p, expected):
    """Test minowski distances."""
    np.testing.assert_array_almost_equal(
        metrics.minowski(array_1, array_2, p),
        expected
    )
