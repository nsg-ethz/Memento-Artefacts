"""Test retraining framework."""
# pylint: disable=redefined-outer-name

from datetime import datetime as dt
from datetime import timedelta

import experiment_helpers as eh
import pandas as pd
import pytest

from experiments import replay_helper
from memento import alternatives, distances, models


@pytest.fixture
def framework():
    """Return a framework implementation."""
    class _Framework(replay_helper.ReplayFramework):
        def __init__(self, iterations, *args, out=None, **kwargs):
            self.iterations = iterations
            self.current_time = dt.now()
            super().__init__(*args, out=out, **kwargs)

        def load_data(self, starting_iteration=0):
            """Return indices as x and y input, 10*indices as eval data.

            Also return index as label.
            """
            for ind in range(starting_iteration, self.iterations):
                yield ({'x': [ind], 'y': [ind], "label": [ind]}, 10 * ind)

        def total_iterations(self):
            return self.iterations

        def evaluate(self, predictor, eval_data):
            """Return a dataframe with predictor and eval data.

            Returns two equal rows.
            """
            return pd.DataFrame.from_records(2*[{
                'predictor': predictor,
                'eval': eval_data,
            }])

        def train(self, traindata, last_predictor=None):
            """Return last predictor + (x * y)."""
            if last_predictor is None:
                return traindata['x'][-1] * traindata['y'][-1]
            return last_predictor + (traindata['x'][-1] * traindata['y'][-1])

        def _now(self):
            """Every time now is called, advance by 1 second."""
            self.current_time += timedelta(seconds=1)
            return self.current_time

    return _Framework


@pytest.fixture(params=["mem", "multimem"])
def memory(request):
    """An infinite memory.

    Alternatively, the memory wrapped in a multi memory, to ensure
    that this works, too.
    """
    memclass = alternatives.InfiniteMemory

    class _MultiMem(models.MultiMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(
                memories=[memclass(*args, **kwargs)]
            )

    return memclass if request.param == "mem" else _MultiMem


@pytest.fixture(params=["mem", "multimem"])
def predictormemory(request):
    """An infinite memory that has a predictor.

    Alternatively, the predictor memory wrapped in a multi memory, to ensure
    that this works, too.
    """
    class _PredictorMem(alternatives.InfiniteMemory):
        def __init__(self, *args, **kwargs):
            self.hist = []
            super().__init__(*args, **kwargs)

        def predict(self, x):
            # Remember the current predictor every time predict is called.
            self.hist.append(self.predictor)
            return x  # Dummy

    class _PredictorMultiMem(models.MultiMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(
                memories=[_PredictorMem(*args, **kwargs)]
            )

    return _PredictorMem if request.param == "mem" else _PredictorMultiMem


def _assert_frames_equal(df1, df2):
    """Assert two dataframes are equal, but allow different column orders."""
    pd.testing.assert_frame_equal(df1.sort_index(axis=1),
                                  df2.sort_index(axis=1))


def test_framework(framework, memory):
    """Test a general framework run."""
    mem = memory()
    runner = framework(3, memory=mem)
    results, stats = runner.run()

    # Testframework returns each row double, and nothign should get lost.
    expected_results = pd.DataFrame({
        'iteration': [0, 0, 1, 1, 2, 2],
        'predictor': [0, 0, 1, 1, 5, 5],  # 2*ind + last
        'eval': [0, 0, 10, 10, 20, 20],
    })
    _assert_frames_equal(results, expected_results)

    expected_stats = pd.DataFrame({
        'iteration': [0, 1, 2],
        'insert_time': [1., 1., 1.],
        'threshold': [0., 0., 0.],
        'interval': [1, 1, 1],
        'retrain': [True, True, True],
    })
    _assert_frames_equal(stats, expected_stats)


def test_framework_predictor_update(framework, predictormemory):
    """Check that the memory predictor is updated."""
    mem = predictormemory(predict_on_update=False, predict_on_insert=True)
    runner = framework(3,
                       memory=mem,
                       train_interval=2,
                       predictor=42)

    if isinstance(mem, models.MultiMemory):
        mem = mem.memories[0]  # test that everything is forwarded to mem.

    assert mem.predictor == 42    # Test initial predictor
    assert runner.predictor == 42

    results, _ = runner.run()

    # We have a predictor at the first iteration, which is passed to train.
    # No new predictor at third iteration.
    expected_results = pd.DataFrame({
        'iteration': [0, 0, 1, 1, 2, 2],
        'predictor': [42, 42,  43, 43, 43, 43],
        'eval': [0, 0, 10, 10, 20, 20],
    })
    _assert_frames_equal(results, expected_results)
    # First predictor used in first two iterations, then new predictor.
    assert mem.hist == [42, 42, 43]
    assert mem.predictor == 43
    assert runner.predictor == 43


def test_output(framework, memory, tmp_path):
    """Test the saving of framwork outputs."""
    runner = framework(3, memory=memory(), out=tmp_path)
    results, stats = runner.run()

    resultfile = tmp_path / "results.csv.gz"
    statfile = tmp_path / "stats.csv.gz"
    assert resultfile.is_file()
    assert statfile.is_file()

    expected_results = pd.DataFrame({
        'iteration': [0, 0, 1, 1, 2, 2],
        'predictor': [0, 0, 1, 1, 5, 5],  # 2*ind + last
        'eval': [0, 0, 10, 10, 20, 20],
    })
    _assert_frames_equal(eh.data.read_csv(resultfile), expected_results)
    _assert_frames_equal(results, expected_results)

    expected_stats = pd.DataFrame({
        'iteration': [0, 1, 2],
        'insert_time': [1., 1., 1.],
        'threshold': [0., 0., 0.],
        'interval': [1, 1, 1],
        'retrain': [True, True, True],
    })
    _assert_frames_equal(eh.data.read_csv(statfile), expected_stats)
    _assert_frames_equal(stats, expected_stats)


def test_output_predictor(framework, memory, tmp_path):
    """Test that a predictor is also saved."""
    runner = framework(3, memory=memory(), out=tmp_path, predictor=42)
    runner.run()
    predictorfile = tmp_path / "predictor"
    assert predictorfile.is_file()
    assert eh.data.read_pickle(predictorfile) == 47  # Final predictor.


def test_output_checkpoint(framework, memory, tmp_path, mocker):
    """Test extended output if checkpoints are enabled."""
    # Pretend we stopped in the middle: do not remove checkpoint at the end.
    mocker.patch.object(framework, "remove_checkpoint")

    pre_runner = framework(2, memory=memory(), out=tmp_path, checkpoints=True)
    pre_runner.run()

    checkpoint = tmp_path / "checkpoint.pickle"
    assert checkpoint.is_file()

    runner = framework(3, memory=memory(), out=tmp_path, checkpoints=True)
    assert runner.starting_iteration == 2  # Checkpoint loaded.
    assert len(list(runner.load_data(2))) == 1  # Double-check test framework.
    results, stats = runner.run()

    # We correctly continue from last results.
    expected_results = pd.DataFrame({
        'iteration': [0, 0, 1, 1, 2, 2],
        'predictor': [0, 0, 1, 1, 5, 5],  # 2*ind + last
        'eval': [0, 0, 10, 10, 20, 20],
    })
    _assert_frames_equal(results, expected_results)

    expected_stats = pd.DataFrame({
        'iteration': [0, 1, 2],
        'insert_time': [1., 1., 1.],
        'threshold': [0., 0., 0.],
        'interval': [1, 1, 1],
        'retrain': [True, True, True],
    })
    _assert_frames_equal(stats, expected_stats)


def test_checkpoint_cleanup(framework, memory, tmp_path):
    """Test that the checkpoint data is removed."""
    runner = framework(3, memory=memory(), out=tmp_path, checkpoints=True)
    runner.run()
    checkpoint = tmp_path / "checkpoint.pickle"
    assert not checkpoint.is_file()


def test_framework_less_retraining(framework, memory):
    """Test retraining every n-th iteration."""
    runner = framework(4, memory=memory(), train_interval=2)
    results, stats = runner.run()

    # No predictor/result at first iteration, no new one at third.
    expected_results = pd.DataFrame({
        'iteration': [1, 1, 2, 2, 3, 3],
        'predictor': [1, 1, 1, 1, 10, 10],
        'eval': [10, 10, 20, 20, 30, 30],
    })
    _assert_frames_equal(results, expected_results)

    expected_stats = pd.DataFrame({
        'iteration': [0, 1, 2, 3],
        'insert_time': [1., 1., 1., 1.],
        'threshold': [0.0, 0.0, 0.0, 0.0],
        'interval': [2, 2, 2, 2],
        'retrain': [False, True, False, True],
    })
    _assert_frames_equal(stats, expected_stats)


def test_framework_final_retraining(framework, memory):
    """Test that we can specify to only train at the very end."""
    runner = framework(4, memory=memory(), train_interval="final")
    results, stats = runner.run()

    # No predictor/result until final iteration (3)
    expected_results = pd.DataFrame({
        'iteration': [3, 3],
        'predictor': [9, 9],
        'eval': [30, 30],
    })
    _assert_frames_equal(results, expected_results)

    expected_stats = pd.DataFrame({
        'iteration': [0, 1, 2, 3],
        'insert_time': [1., 1., 1., 1.],
        'threshold': [0.0, 0.0, 0.0, 0.0],
        'interval': [4, 4, 4, 4],
        'retrain': [False, False, False, True],
    })
    _assert_frames_equal(stats, expected_stats)


@pytest.mark.parametrize("mem_filter, expected", [
    (None, [21, 42]),
    ([0], [21]),
    ([1], [42]),
])
def test_multimem_filter(framework, mem_filter, expected):
    """Test using only specific memories from a Multimemory."""
    # pylint: disable=protected-access
    data1 = {'somekey': [21]}
    data2 = {'somekey': [42]}

    mem = models.MultiMemory(
        memories=[alternatives.InfiniteMemory(), alternatives.InfiniteMemory()]
    )
    mem.data = (data1, data2)

    runner = framework(1, memory=mem, multi_mem_filter=mem_filter)
    output = runner._comparison_data(mem.data)
    assert set(output.keys()) == {'somekey'}
    assert output['somekey'].tolist() == expected


def test_coverage_increase(framework):
    """Test that via memento, the framework can check for coverage increase."""
    class DummyPredictorMemento(models.Memento):
        """Work with the dummy predictor updates from the framework."""

        def predict(self, x):
            return x

    mem = DummyPredictorMemento(distances=[
        distances.JSDCategoricalPoint('x', classes=100),
        distances.JSDCategoricalPoint('y', classes=100),
        distances.JSDCategoricalPoint('yhat', classes=100),
    ], batching_trim=False)

    # We need to specify a train threshold, otherwise the framework will
    # silently ignore errors as the coverage increase is not needed.
    runner = framework(4, memory=mem, train_threshold=0.1)
    _, stats = runner.run()

    # Test that the stats columsn exist, and the first row is NaN.
    # (can't compute change in first step)
    stat_cols = ['coverage_increase', 'total_coverage_increase',
                 'total_coverage']
    assert all(stats.loc[0, stat_cols].isna())

    # All other values should be positive
    assert all(stats[stat_cols][1:] >= 0.0)

    # Additional bounds on the (relative) coverage increase.
    assert all(stats['coverage_increase'][1:] <= 1.0)
    # We expect some increase during the test run.
    assert any(stats['coverage_increase'][1:] > 0.0)
