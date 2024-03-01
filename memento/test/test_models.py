"""Test replay memory models."""
# pylint: disable=too-many-ancestors, invalid-name, redefined-outer-name

from functools import partial

import numpy as np
import pytest

from memento import bases, models, utils


def _tosets(datadict):
    """Convert arrays to lists for easier comparison."""
    return {k: set(v) for k, v in datadict.items()}


def distance1(batches, others=None):
    """First dummy distance, highest to lowest density: 2, 1, 0, 3."""
    assert len(batches) == 4, "Works only for exactly 4 batches."
    assert others is None, "Only works for pairwise distances."
    return np.array([
        [0.0, 0.2, 0.1, 1.0],
        [0.2, 0.0, 0.1, 0.2],
        [0.1, 0.1, 0.0, 0.1],
        [1.0, 0.2, 0.1, 0.0]
    ])


def distance2(batches, others=None):
    """Second dummy distance, highest to lowest density: 1, 0, 3, 2.

    Generally higher distances/lower density than distance1.
    """
    assert len(batches) == 4, "Works only for exactly 4 batches."
    assert others is None, "Only works for pairwise distances."
    return np.array([
        [0.0, 0.2, 1.0, 2.0],
        [0.2, 0.0, 1.0, 0.2],
        [1.0, 1.0, 0.0, 2.0],
        [2.0, 0.2, 2.0, 0.0]
    ])


def mock_distance(batches, others=None):
    """Returns a random matrix."""
    n_rows = len(batches)
    n_cols = len(batches) if others is None else len(others)
    return np.random.rand(n_rows, n_cols)


@pytest.fixture()
def testmemory():
    """Test memory class fixture."""
    class _TestMemory(models.Memento):
        def __init__(self, *args, bw=0.5, distances=None, batching_size=1, **kwargs):
            if distances is None:
                distances = [distance1, distance2]
            super().__init__(*args, distances=distances, bw=bw,
                             temperature=0.0,
                             batching_size=batching_size, **kwargs)

        def kernel(self, x):
            """Disable kernel with bw=None for easier testing."""
            if self.bw is not None:
                return super().kernel(x)
            else:
                return x

    return _TestMemory


def test_kernel(testmemory):
    """Test kernel and bandwidth."""
    mem1 = testmemory(bw=0.1)
    mem2 = testmemory(bw=0.2)
    assert mem1.kernel(0.1) == mem2.kernel(0.2)  # Correct scaling.
    # Test different input shapes while we are at it.
    assert np.array_equal(mem1.kernel(0.1 * np.ones((3, 3))),
                          mem2.kernel(0.2 * np.ones((3, 3))))


@pytest.mark.parametrize('size, expected', [
    (4, {'x': {0, 1, 2, 3}}),
    (3, {'x': {0, 2, 3}}),
    (2, {'x': {0, 3}}),
    (1, {'x': {0}}),  # Two elements -> equal density; last one is removed.
    (0, {}),
])
def test_selection_size(testmemory, size, expected):
    """Test different memory sizes."""
    testmem = testmemory()
    testmem.insert(x=np.arange(4))
    assert utils.sample_count(testmem.data) == 4
    assert _tosets(testmem.get(size)) == expected

    testmem = testmemory(size=size)
    testmem.insert(x=np.arange(4))
    assert utils.sample_count(testmem.data) == size
    assert _tosets(testmem.get()) == expected


def test_multi_samples(testmemory):
    """Test that <= max samples are selected if batchsize does not match."""
    # Use uneven sample sizes: 0 and 2 have 6 samples each.
    mem = testmemory(bw=0.5, size=3, batching_size=2)
    mem.insert(x=np.arange(8))
    output = mem.get()
    assert utils.sample_count(output) == 2


@pytest.mark.parametrize("distances, expected", [
    ([distance1, distance2], {0, 2, 3}),
    ([distance1], {1, 0, 3}),
    ([distance2], {0, 3, 2}),
    ([], {0, 1, 2})  # FIFO Fallback.
])
def test_distance_combinations(testmemory, distances, expected):
    """Test sample selection for different distance combinations.

    Each combinations has a different highest-density batch.
    """
    mem = testmemory(bw=0.5, size=3, distances=distances)
    mem.insert(x=np.arange(4))
    assert _tosets(mem.get()) == {'x': expected}


def test_insert_chunked(testmemory, mocker):
    """Test that data can be chunked to limit memory overhead when inserting."""
    mock = bases.MemoryBase.insert_datadict = mocker.Mock()
    mem = testmemory(insert_chunksize=2)

    mem.insert(x=np.arange(4))
    for call, array in zip(mock.mock_calls, ([0, 1], [2, 3])):
        assert len(call.args) == 1
        assert set(call.args[0].keys()) == {'x'}
        np.testing.assert_array_equal(call.args[0]['x'], array)



def test_distance_samples(testmemory):
    """Test that we can compute the distance using a subset of batches.
    
    Only test that it doesn't crash.
    """
    mem = testmemory(
        distances=[mock_distance],
        distance_sample_fraction=0.01,
        size=32,
    )
    mem.insert(x=np.arange(100))



# Multi memory tests.
# ===================

@pytest.fixture()
def testmultimemory():
    """Test multi memory class fixture.

    Creates two dummy memories with dummy batches.
    The first dummy accepts a predictor.
    """
    class _DummyOne(bases.MemoryBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, size=1, **kwargs)
            self.data = {'x': np.array([1, 2])}
            self.predictor = None

        def select(self, max_samples, current_data, new_data):
            return new_data

        def predict(self, x):
            return x

    class _DummyTwo(bases.MemoryBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, size=1, **kwargs)
            self.data = {'x': np.array([3, 4])}

        def select(self, max_samples, current_data, new_data):
            return new_data

    return partial(models.MultiMemory, memories=[_DummyOne(), _DummyTwo()])


@pytest.mark.parametrize("weights, expected", [
    (None, [0.5, 0.5]),
    ([0.7, 0.3], [0.7, 0.3]),
    ([3, 1], [0.75, 0.25])
])
def test_multimem_init(testmultimemory, weights, expected):
    """Test weight initialization."""
    mem = testmultimemory(weights=weights)
    assert mem.weights == expected


def test_multimem_data(testmultimemory):
    """Test accessing `.data` returns a tuple of memory data."""
    data = testmultimemory().data
    assert len(data) == 2
    assert _tosets(data[0]) == {'x': {1, 2}}
    assert _tosets(data[1]) == {'x': {3, 4}}


def test_multimem_data_overwrite(testmultimemory):
    """Test passing a batch tuple overwrites memory batches directly."""
    mem = testmultimemory()
    mem.data = ({'x': np.array([5, 6])}, {'x': np.array([7])})
    assert _tosets(mem.memories[0].data) == {'x': {5, 6}}
    assert _tosets(mem.memories[1].data) == {'x': {7, }}


def test_multimem_insert(testmultimemory, mocker):
    """Test that all memories get data."""
    mocks = [mocker.Mock(), mocker.Mock()]
    mockarg = mocker.Mock()
    mem = testmultimemory(memories=mocks)
    mem.insert(mockarg)
    for mock in mocks:
        mock.insert.assert_called_once_with(mockarg)


def test_multimem_get(testmultimemory):
    """Test that data from all memories is merged."""
    assert _tosets(testmultimemory().get()) == {'x': {1, 2, 3, 4}}


@pytest.mark.parametrize("weights, size, expected", [
    (None, 1000, [500, 500]),
    ([0.7, 0.3], 100, [70, 30]),
    ([3, 1], 4, [3, 1])
])
def test_multimem_get_maxsize(testmultimemory, mocker,
                              weights, size, expected):
    """Test that maxsize is multiplied by weights."""
    mocks = [mocker.Mock(), mocker.Mock()]
    for mock in mocks:
        mock.get = mocker.Mock(return_value={"x": [42]})
    mem = testmultimemory(memories=mocks, weights=weights)
    mem.get(max_samples=size)
    for mock, _size in zip(mocks, expected):
        mock.get.assert_called_once_with(max_samples=_size)


def test_multimem_update_predictor(testmultimemory):
    """Test that `update_predictor` is called for memories."""
    mem = testmultimemory()
    mem.data = ({}, {})  # Don't need to test repredicting again.
    mem.update_predictor(42)
    assert mem.memories[0].predictor == 42
    assert mem.memories[1].predictor == 42


def test_multimem_attr(testmultimemory):
    """Test accessing attributes is delegated to memories."""
    mem = testmultimemory()
    mem.blabla = 42
    assert mem.blabla == (42, 42)
    for _mem in mem.memories:
        assert _mem.blabla == 42
    mem.blabla = 21
    assert mem.blabla == (21, 21)
    for _mem in mem.memories:
        assert _mem.blabla == 21


# Batching tests.
# ===============

@pytest.mark.parametrize('_x', [None, np.arange(1000)[:, np.newaxis]])
@pytest.mark.parametrize('_y', [None, np.arange(1000) + 10])
@pytest.mark.parametrize('_yhat', [None, np.arange(1000) + 100])
@pytest.mark.parametrize('batchsize', [1, 43, 1000, 1001])
def test_batch_notrim(testmemory, _x, _y, _yhat, batchsize):
    """By befault, each row is converted into a single batch."""
    mem = testmemory(batching_size=batchsize, batching_trim=False)
    data = {}
    for var, label in ((_x, 'x'), (_y, 'y'), (_yhat, 'yhat')):
        if var is not None:
            data[label] = var

    batches = mem.batch(data)

    if all(item is None for item in (_x, _y, _yhat)):
        assert batches == []
        return

    # Note: We use inputs with 1000 elements.
    assert len(batches) == int(np.ceil(1000 / batchsize))
    assert sum(utils.sample_count(batch) for batch in batches) == 1000
    for batch in batches[:-1]:  # Only the last batch may be trimmed.
        assert utils.sample_count(batch) == batchsize

    merged = utils.merge_datadicts(batches)
    for var, label in ((_x, 'x'), (_y, 'y'), (_yhat, 'yhat')):
        if var is None:
            assert label not in merged
        else:
            np.testing.assert_array_equal(merged[label], var)


@pytest.mark.parametrize("attr", ["x", "y", "yhat"])
def test_add_data_trim(testmemory, attr):
    """Test data trimming for even-sized batches."""
    mem = testmemory(batching_size=101, batching_trim=True)
    data = np.arange(1000)
    batches = mem.batch({attr: data})

    assert len(batches) == 9
    merged = utils.merge_datadicts(batches)
    assert utils.sample_count(merged) == 909
    np.testing.assert_array_equal(merged[attr], np.arange(909))


@pytest.mark.parametrize("attr", ["x", "y", "yhat"])
def test_batching_sort(testmemory, attr):
    """Test sorting when batching."""
    mem = testmemory(batching_size=2, batching_sort=(attr,))
    data = np.array([0, 0, 0, 3, 2, 3, 1, 1])
    expected = [[0, 0], [0, 1], [1, 2], [3, 3]]

    batches = mem.batch({attr: data})
    for batchdata, expected_data in zip(batches, expected):
        np.testing.assert_array_equal(batchdata[attr], expected_data)


def test_batching_sort_multi(testmemory):
    """Test that we sort in the specified order."""
    expected = np.array([
        # Even though the latter entries are higher, sort by previous cols.
        # y, yhat, x
        [0, 10, 10],
        [1, 0, 10],
        [1, 1, 0],
        [1, 1, 1],
    ])
    reverse = expected[::-1]  # So everything needs sorting.
    _y = reverse[:, 0]
    _yhat = reverse[:, 1]
    _x = reverse[:, 2]

    mem = testmemory(batching_size=1, batching_sort=('y', 'yhat', 'x'))
    batches = mem.batch(dict(x=_x, y=_y, yhat=_yhat))
    for batch, row in zip(batches, expected):
        np.testing.assert_array_equal(batch['y'], row[0:1])
        np.testing.assert_array_equal(batch['yhat'], row[1:2])
        np.testing.assert_array_equal(batch['x'], row[2:])


@pytest.mark.parametrize("attr", ["x", "y", "yhat"])
def test_add_data_sort_2d_max(testmemory, attr):
    """Method 'max' sorts 2d data by argmax and max.

    Mostly intended for distributions: Sort by mode, then probability of mode.
    """
    # Start with correctly sorted array.
    expected = np.array([
        # Argmax: 0, narrow to wide.
        [1, 0, 0],
        [0.5, 0.3, 0.2],
        # Argmax: 1 (first index).
        [0, 0.5, 0.5],
        # Argmax: 2.
        [0, 0, 1],
    ])
    reverse = expected[::-1]  # So everything needs sorting

    mem = testmemory(batching_size=1, batching_sort=(attr,))
    batches = mem.batch({attr: reverse})
    for batch, row in zip(batches, expected):
        np.testing.assert_array_equal(batch[attr].squeeze(), row)


# Tests for subsampling (and sparse arrays to make it feasible).
# =============================================================

def test_memento_sample_fraction(testmemory):
    """Simply test that Memento doesn't crash when using sparse arrays."""
    mem = testmemory(size=100,
                     distances=[mock_distance],
                     distance_sample_fraction=0.5)
    mem.insert(x=np.arange(1000))
