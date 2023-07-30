"""Test alternative replay memory algorithms."""

import numpy as np
import pytest

from memento import alternatives, utils


@pytest.mark.parametrize('kwargs, key', [
    # Whether the sample start in memory or not, the same output is expected.
    (dict(), 'random'),
    (dict(key='key'), 'key')
])
def test_reservoir_draw(kwargs, key):
    """When inserting samples, they get random values."""
    testmem = alternatives.ReservoirMemory(**kwargs)
    testmem.insert(x=[10, 20])
    output = testmem.get()
    assert key in output
    assert isinstance(output[key], np.ndarray)
    assert np.isreal(output[key]).all()


@pytest.mark.parametrize('kwargs, key', [
    # Whether the sample start in memory or not, the same output is expected.
    (dict(), 'random'),
    (dict(key='key'), 'key')
])
def test_reservoir_select(kwargs, key):
    """Elements are selected by the pre-drawn keys."""
    testmem = alternatives.ReservoirMemory(**kwargs)
    # it does not matter whether new or old.
    output1 = testmem.select(2, {'x': [1, 2, 3], key: [0.5, 0, 1]}, {})
    output2 = testmem.select(2, {}, {'x': [1, 2, 3], key: [0.5, 0, 1]})
    output3 = testmem.select(2, {'x': [1], key: [0.5]},
                             {'x': [2, 3], key: [0, 1]})
    for output in (output1, output2, output3):
        assert list(output[key]) == [0, 0.5]
        assert list(output['x']) == [2, 1]


def test_reservoir_forget():
    """Test reservoir memory."""
    testmem = alternatives.ReservoirMemory(size=1000, forget=0.5)
    testmem.insert(x=[1] * 1000)
    testmem.insert(x=[2] * 500)
    output = testmem.get()
    # Because half of the batches are forgotten, all new batches are in.
    assert utils.sample_count(output) == 1000
    assert len([v for v in output['x'] if v == 1]) == 500
    assert len([v for v in output['x'] if v == 2]) == 500


def test_fifo():
    """Test reservoir memory."""
    testmem = alternatives.FIFOMemory(size=4)
    testmem.insert(x=[1, 2, 3])
    testmem.insert(x=[4, 5, 6])
    assert list(testmem.get()['x']) == [3, 4, 5, 6]


def test_dummies():
    """Test the dummy memory that can be initialized."""
    mem = alternatives.NoMemory(size=1)
    mem.insert(x=list(range(10)))
    assert mem.get() == {}  # Takes nothing.
    mem = alternatives.InfiniteMemory(size=1)
    mem.insert(x=list(range(10)))
    assert list(mem.get()['x']) == list(range(10))  # disregards size.
