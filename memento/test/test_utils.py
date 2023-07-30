"""Test utils."""

import ctypes
import multiprocessing as mp

import numpy as np
import pytest

from memento import utils


def _echo1(arg):
    return arg


def _echo2(arg1, arg2):
    return (arg1, arg2)


def _echo_shared(index):
    """Each index is one element in array"""
    return np.frombuffer(utils.SHARED_ARRAYS)[index]


def _echo_shared_multi(index):
    """Each index is one array."""
    return list(np.frombuffer(utils.SHARED_ARRAYS[index]))


@pytest.mark.parametrize("workers", [1, 2, 100, None])
def test_multiprocessing_map(workers):
    """Test the map function of the parallel class for varying workers."""
    data = list(range(1000))
    parallel = utils.MultiProcessing(workers=workers)
    assert parallel.map(_echo1, data) == data


@pytest.mark.parametrize("workers", [1, 2, 100, None])
def test_multiprocessing_starmap(workers):
    """Test the map function of the parallel class for varying workers."""
    data = list(zip(range(1000), range(1, 1001)))
    parallel = utils.MultiProcessing(workers=workers)
    assert parallel.starmap(_echo2, data) == data


@pytest.mark.parametrize("workers", [1, 2, 100, None])
def test_multiprocessing_shared_array(workers):
    """Test sharing a single array."""
    parallel = utils.MultiProcessing(workers=workers)
    arr = mp.RawArray(ctypes.c_double, 5)
    data = [42, 21, 1, 2, 3]

    # Use numpy to fill array.
    arrnp = np.frombuffer(arr)
    arrnp[:] = data
    np.testing.assert_allclose(arrnp, data)

    # Reach each index of single shared array
    res = parallel.map(_echo_shared, list(range(5)), shared_arrays=arr)
    assert res == data


@pytest.mark.parametrize("workers", [1, 2, 100, None])
def test_multiprocessing_shared_arrays(workers):
    """Test sharing multiple arrays."""
    parallel = utils.MultiProcessing(workers=workers)
    data = [42, 21, 1, 2, 3]
    arr1 = mp.RawArray(ctypes.c_double, 2)
    arr2 = mp.RawArray(ctypes.c_double, 3)

    # Use numpy to fill array.
    arr1np = np.frombuffer(arr1)
    arr1np[:] = data[:2]
    arr2np = np.frombuffer(arr2)
    arr2np[:] = data[2:]
    np.testing.assert_allclose(arr1np, [42, 21])
    np.testing.assert_allclose(arr2np, [1, 2, 3])

    # Reach each index of single shared array
    res = parallel.map(_echo_shared_multi, [0, 1], shared_arrays=[arr1, arr2])
    assert res == [[42, 21], [1, 2, 3]]


def test_check_datadict():
    """Arraylike are converted to arrays, arrays stay the same."""
    _input = dict(x=[1, 2], y=np.array([3, 4]))
    _output = utils.check_datadict(_input)

    assert list(_input.keys()) == ['x', 'y']
    for key in 'x', 'y':
        assert isinstance(_output[key], np.ndarray)
        np.testing.assert_equal(_output[key], _input[key])


def test_check_datadict_scalar():
    """Scalar inputs are converted to 1d arrays."""
    _input = dict(x=1, y=np.array(3))
    _output = utils.check_datadict(_input)

    assert list(_input.keys()) == ['x', 'y']
    for key in 'x', 'y':
        assert isinstance(_output[key], np.ndarray)
        assert _output[key].ndim == 1
        np.testing.assert_equal(_output[key], [_input[key]])


@pytest.mark.parametrize("data", [
    {},
    {'empty': []},
    {'alsoempty': None},
    {'thistoo': np.array([])},
])
def test_check_datadict_empty(data):
    """And empty dict is passed through."""
    assert utils.check_datadict(data) == {}


@pytest.mark.parametrize("data", [
    1,
    None,
    [None],
    [[1, 2]],
    {'x': [1], 'y': [1, 2]},
])
def test_check_datadict_bad_format(data):
    """We need a dict."""
    with pytest.raises(Exception):
        utils.check_datadict(data)


@pytest.mark.parametrize("data, expected", [
    ({}, 0),
    (dict(x=[1, 2], y=[3, 4]), 2),
])
def test_check_sample_count(data, expected):
    """Getting the length of data in a datadict."""
    assert utils.sample_count(data) == expected


@pytest.mark.parametrize("data_a, data_b, expected", [
    # Basics:
    ({}, {}, 0),
    ({'x': [1]}, {}, 1),
    ({'x': [1]}, {'x': [2]}, 1),
    ({'x': [1, 2]}, {}, 2),
    ({'x': [1, 2, 3]}, {'x': [2]}, 2),
    ({'x': [1, 2]}, {'x': [2, 1]}, 0),
    ({'x': [2, 2]}, {'x': [2]}, 1),
    ({'x': [2, 2, 2, 2]}, {'x': [2, 2]}, 2),
    # Test multi data rows
    ({'x': [0], 'y': [0]}, {'x': [1], 'y': [1]}, 1),
    ({'x': [[0], [1]], 'y': [0, 1]}, {'x': [[1]], 'y': [1]}, 1),
    # Corner cases: mis-matched keys
    ({'x': [1]}, {'x': [1]}, 0),
    ({'x': [1]}, {'y': [1]}, 1),
    # Corner case: weird data-type combinations.
    ({'x': np.int64([42]), 'y': [np.datetime64('1991-07-27')]},
     {'x': np.int64([42]), 'y': [np.datetime64('1991-07-27')]}, 0),
    ({'x': np.int64([42]), 'y': [np.datetime64('1991-07-27')]},
     {'x': np.int64([21]), 'y': [np.datetime64('1991-07-21')]}, 1)
])
def test_sample_diff(data_a, data_b, expected):
    """Test computing a sample diff between batches."""
    # Test result and symmetry.
    data_a = utils.check_datadict(data_a)
    data_b = utils.check_datadict(data_b)
    assert utils.sample_diff(data_a, data_b) == expected
    assert utils.sample_diff(data_b, data_a) == expected


@ pytest.mark.parametrize("dicts, expected", [
    (({}, {}), {}),
    (({'x': [1]}, {}), {'x': [1]}),
    (({'x': [1]}, {'x': [2]}), {"x": [1, 2]}),
    (({'x': [1, 2], 'y': [3, 4]}, {'x': [5], 'y': [6]}),
     {"x": [1, 2, 5], 'y': [3, 4, 6]}),
    (({'x': [[1, 1], [2, 2]], 'y': [3, 4]}, {'x': [[5, 5]], 'y': [6]}),
     {"x": [[1, 1], [2, 2], [5, 5]], 'y': [3, 4, 6]}),
    # Only shared keys can be merged.
    (({'x': [1]}, {'y': [2]}), {}),
    (({'x': [1], 'y': [2]}, {'y': [3]}), {'y': [2, 3]})
])
def test_merge_datadicts(dicts, expected):
    """Test merging datadicts"""
    # Merging expects checked datadicts.
    checked = [utils.check_datadict(_d) for _d in dicts]
    merged = utils.merge_datadicts(checked)
    for key, value in expected.items():
        np.testing.assert_array_equal(value, merged[key])


@ pytest.mark.parametrize("array, expected_index", [
    ([0], 0),
    ([0, 0], 1),
    ([1, 0, 0], 0),
    ([1, 0, 0, 1], 3),
])
def test_last_argmax(array, expected_index):
    """Return the index of the last highest element."""
    assert utils.last_argmax(array) == expected_index
