"""Test base class utilities."""
# pylint: disable=redefined-outer-name
# (because of fixtures)

import numpy as np
import pytest

from memento import bases, utils


def _tolists(datadict):
    """Convert arrays to lists for easier comparison."""
    return {k: list(v) for k, v in datadict.items()}


@pytest.fixture
def testbase():
    """Simple fifo memory."""
    class _TestMemoryBase(bases.MemoryBase):
        def select(self, max_samples, current_data, new_data):
            """Return the last batches and log call.

            Assumes that each batch contains one sample of length 1.
            """
            merged = utils.merge_datadicts([current_data, new_data])
            if utils.sample_count(merged) > max_samples:
                ind = np.arange(max_samples)
                return {k: v[ind] for k, v in merged.items()}
            return merged

        def predict(self, x):
            """Simply mirror x data and add the "predictor"."""
            return self.predictor + x
    return _TestMemoryBase


@pytest.mark.parametrize("size, expected", [
    (None, [1, 2]),  # Infinite.
    (1, [1]),        # Finite.
    (3, [1, 2]),     # Finite, but fewer batches than limit.
])
def test_insert(testbase, size, expected):
    """Add batches with different size limitations and formats.

    We use a sample [1, 2] for all tests.
    """
    # The following input formats should all be equivalent
    inputs = [
        # (args, kwargs)
        ([], {"x": [1, 2], 'y': [1, 2]}),
        ([{"x": [1, 2], 'y': [1, 2]}], {}),
        ([{"x": [1], 'y': [1]}, {"x": [2], 'y': [2]}], {}),
        ([{"x": [1], 'y': [1]}], {"x": [2], 'y': [2]}),
        ([{"x": 1, 'y': 1}, {"x": 2, 'y': 2}], {}),
        ([{"x": 1, 'y': 1}], {"x": 2, 'y': 2}),
    ]

    for args, kwargs in inputs:
        mem = testbase(size=size)
        assert not mem.require_predictor
        mem.insert(*args, **kwargs)
        assert _tolists(mem.get()) == {'x': expected, 'y': expected}


@pytest.mark.parametrize("size, expected", [
    (None, [1, 2]),  # Infinite.
    (1, [1]),        # Finite.
    (3, [1, 2]),     # Finite, but fewer batches than limit.
])
def test_mem_get(testbase, size, expected):
    """Get batches with different size limitations."""
    mem = testbase(size=None)
    mem.insert(x=[1, 2], y=[1, 2])
    assert _tolists(mem.get(size)) == {'x': expected, 'y': expected}


@pytest.mark.parametrize("samples, predictions", [
    # We use a "predictor" of value 42, so our test model just adds 42 to x.
    ({'x': 1}, {'yhat': [43]}),
    ({'x': [1]}, {'yhat': [43]}),
    ({'x': [1], 'doesntmatter': [2]}, {'yhat': [43]}),
    ({'x': [1, 2]}, {'yhat': [43, 44]}),
])
def test_predict_on_insert(testbase, samples, predictions):
    """When batches are added, missing predictions are added."""
    mem = testbase(predictor=42, predict_on_insert=True)
    mem.insert(samples)
    output = _tolists(mem.get())

    for key, expected in predictions.items():
        assert output[key] == expected


def test_insert_no_predict(testbase):
    """We can explicitly turn off predictions when adding."""
    mem = testbase(predictor=42, predict_on_insert=False)
    mem.insert(x=1, yhat=2)
    assert _tolists(mem.get()) == {'x': [1], 'yhat': [2]}


def test_require_predictor(testbase):
    """If required, raise RuntimeError for missing predictor."""
    mem = testbase(predictor=None, require_predictor=True,
                   predict_on_insert=True)
    with pytest.raises(RuntimeError):
        mem.insert(x=1)


def test_update_predictor(testbase):
    """Updating the predictor can update predictions."""
    mem = testbase(predict_on_update=True)
    mem.insert(x=1)
    assert _tolists(mem.get()) == {'x': [1]}
    mem.update_predictor(42)
    assert _tolists(mem.get()) == {'x': [1], 'yhat': [43]}


def test_update_predictor_no_predict(testbase):
    """Updating predictions can be turned off."""
    mem = testbase(predict_on_update=False)
    mem.insert(x=1)
    assert _tolists(mem.get()) == {'x': [1]}
    mem.update_predictor(42)
    assert _tolists(mem.get()) == {'x': [1]}


def test_update_predictor_no_samples(testbase):
    """When batches are added, missing predictions are added."""
    mem = testbase(predict_on_update=True)  # Empty memory.
    mem.update_predictor(42)
    assert mem.get() == {}
