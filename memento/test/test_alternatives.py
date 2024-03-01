"""Test alternative replay memory algorithms."""

import numpy as np
from numpy.core.multiarray import array as array
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


class DummyQBC(alternatives.QueryByCommitteeMemory):
    """QBC memory where predictors are just numbers and output is fixed.
    
    Both commitee and outputs are provided by p_map.
    """
    def __init__(self, p_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_map = p_map
        if self.p_map:
            self.predictor = list(self.p_map.keys())
            self.predictor_members = len(self.predictor)

    def predict_probabilities(self, predictor, x) -> np.array:
        return np.array(self.p_map[predictor])


def test_qbc_predictor_update():
    """Test that we keep the latest predictors."""
    model = DummyQBC({}, committee_size=3)
    assert model.predictor == None
    model.update_predictor(1)
    assert model.predictor == [1]
    model.update_predictor(2)
    assert model.predictor == [1, 2]
    model.update_predictor(3)
    assert model.predictor == [1, 2, 3]
    model.update_predictor(4)
    assert model.predictor == [2, 3, 4]


def test_qbc_predictor_update_multiple():
    """Test that we keep the latest predictors."""
    model = DummyQBC({}, committee_size=3)
    model.update_predictors([1, 2, 3, 4])
    assert model.predictor == [2, 3, 4]


def test_qbc_entropy():
    """Test that we keep the samples with highest disagreement/entropy."""
    # Three predictors, four samples.
    # Samples (mixed order) with three predictions each.
    # full agreement, full uncertainty, partial agreement, full disagreement
    model = DummyQBC({
        1: [[1.0, 0., 0.], [1/3, 1/3, 1/3], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        2: [[1.0, 0., 0.], [1/3, 1/3, 1/3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        3: [[1.0, 0., 0.], [1/3, 1/3, 1/3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    })
    probs = model.predict([1, 2, 3, 4])
    entropy = model.entropy(probs)
    assert entropy[0] == min(entropy)  # full agreement
    assert entropy[1] == max(entropy)  # full uncertainty
    assert entropy[3] == max(entropy)  # full disagreement
    assert entropy[0] < entropy[2] < entropy[3]  # partial agreement


@pytest.mark.parametrize('size, selected', [
    (2, {1, 3}),
    (3, {1, 2, 3}),
    (4, {0, 1, 2, 3}),
])
def test_qbc_select(size, selected):
    """Same samples as in the entropy test, check that selection works."""
    model = DummyQBC({
        1: [[1.0, 0., 0.], [1/3, 1/3, 1/3], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        2: [[1.0, 0., 0.], [1/3, 1/3, 1/3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        3: [[1.0, 0., 0.], [1/3, 1/3, 1/3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }, size=size)
    model.insert(x=[0, 1, 2, 3])
    assert set(model.data['x']) == selected
