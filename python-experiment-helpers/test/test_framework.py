"""Test the framework utility functions."""
# pylint: disable=too-few-public-methods, redefined-outer-name

import logging
import shutil
import sys
from functools import partial
from time import sleep

import cloudpickle
import pytest
from experiment_helpers import framework
from experiment_helpers.config import BaseConfig
from experiment_helpers.data import Path


class TestConfig(BaseConfig):
    attribute = 21
    loglevel = logging.DEBUG


def _write(path, content, mkdir=True):
    if mkdir:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:  # pylint: disable=unspecified-encoding
        file.write(str(content))


def _assert_content(path, content):
    with open(path) as file:  # pylint: disable=unspecified-encoding
        assert file.read() == str(content)


def experimentfunc(config=None, arg: str = "test"):
    """Simply write a file using containing a config key as content."""
    # Just write to a file in the current dir, don't try to create dirs.
    TestConfig.with_updates(config)
    _write(arg, config.attribute, mkdir=False)


def noop():
    """Take no config, do nothing."""


def test_init_parametrize():
    """We only can use leaves as experiment names."""
    # Test that this raises an exception.
    with pytest.raises(AssertionError):
        framework.ParametrizedExperiments("experiments", {
            "dir": experimentfunc,
            "dir/leaf": experimentfunc,
        })

    # This does not raise:
    framework.ParametrizedExperiments("experiments", {
        "dir/leaf_a": experimentfunc,
        "dir/leaf_b": experimentfunc,
    })


@pytest.fixture
def parametrized():
    """Fixtrue for parametrized tests."""
    experiments = {
        'leaf': partial(experimentfunc, arg="results.txt"),
        'dir/leaf': partial(experimentfunc, arg="filename"),
        'dir/nooutput': noop,  # noop.
    }
    parametrized = framework.ParametrizedExperiments(
        "experiments", experiments, TestConfig)

    return parametrized


@pytest.mark.parametrize('config_overrides', [
    {},                            # default
    dict(multiprocessing_jobs=2),  # multiprocessing
])
def test_parametrize(parametrized, tmp_path, config_overrides):
    """Test parametrization."""
    parametrized(config=dict(
        output_directory=str(tmp_path),
        **config_overrides,
    ))
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 21)
    _assert_content(tmp_path / 'experiments/dir/leaf/filename', 21)


def test_parametrize_slurm(parametrized, request):
    """Test parametrization with SLURM. Fails if not aviailable.

    We need to use a output directory that is mapped to the SLURM nodes.
    In our setup, this is true for the user home directory.
    """
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    tmp_path = Path(f"./.tmp_slurmresults_{request.node.name}")
    assert not tmp_path.exists()
    try:
        tmp_path.mkdir(parents=True, exist_ok=True)
        parametrized(config=dict(
            output_directory=str(tmp_path),
            slurm=True,
        ))
        # We can have slight synchronization delays, so wait a bit.
        sleep(1.0)
        _assert_content(tmp_path / 'experiments/leaf/results.txt', 21)
        _assert_content(tmp_path / 'experiments/dir/leaf/filename', 21)
    finally:
        # Remove the tmpdir.
        shutil.rmtree(tmp_path)


def test_parametrize_pattern(parametrized, tmp_path):
    """Test parametrization and call with pattern."""
    parametrized(
        "dir/le*",  # Excludes other experiment.
        config=dict(output_directory=str(tmp_path))
    )
    _assert_content(tmp_path / 'experiments/dir/leaf/filename', 21)
    assert not Path(tmp_path / 'experiments/leaf/results.txt').exists()


def test_multi_pattern(parametrized, tmp_path):
    """Test parametrization with multiple patterns."""
    parametrized(
        ["leaf", "dir/leaf"],  # match everything explicitly.
        config=dict(output_directory=str(tmp_path))
    )
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 21)
    _assert_content(tmp_path / 'experiments/dir/leaf/filename', 21)


@pytest.mark.parametrize('pattern', ["dir", "dir/"])
def test_sub_pattern(parametrized, tmp_path, pattern):
    """Test parametrization with a directory that has children."""
    parametrized(pattern, config=dict(output_directory=str(tmp_path)))
    # Should match everything within dir.
    assert not Path(tmp_path / 'experiments/leaf/results.txt').exists()
    _assert_content(tmp_path / 'experiments/dir/leaf/filename', 21)


def test_parametrize_skip(parametrized, tmp_path):
    """Test that tests with existing output dirs with files are skipped."""
    _write(tmp_path / 'experiments/leaf/results.txt', 123)
    # Create empty dir, should not be skipped.
    (tmp_path / 'experiment/dir/subsubdir').mkdir(parents=True)

    parametrized(config=dict(output_directory=str(tmp_path)))
    # Unchanged, i.e. function did not run.
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 123)
    _assert_content(tmp_path / 'experiments/dir/leaf/filename', 21)


def test_parametrize_dont_skip_checkpoints(parametrized, tmp_path):
    """Test that dirs containing checkpoints are not skipped."""
    # Results.
    _write(tmp_path / 'experiments/leaf/results.txt', 123)
    # But also a checkpoint.
    _write(tmp_path / 'experiments/leaf/checkpoint', "anything")

    parametrized(config=dict(
        output_directory=str(tmp_path),
        checkpoint_files=('checkpoint',),
    ))
    # `leaf` should not be skipped, as a checkpoint exists.
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 21)
    # The checkpoint file should still exist, i.e., no forced cleanup.
    _assert_content(tmp_path / 'experiments/leaf/checkpoint', "anything")


def test_ignore_files(parametrized, tmp_path):
    """Test that dirs with ignored files are skipped."""
    # Results and something to ignore
    _write(tmp_path / 'experiments/leaf/results.txt', 123)
    _write(tmp_path / 'experiments/leaf/ignore', "anything")

    parametrized(config=dict(
        output_directory=str(tmp_path),
        ignore_files=['ignore',],
    ))
    # `leaf` should be skipped.
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 123)


def test_only_ignored_files(parametrized, tmp_path):
    """Test that dirs containing only ignored files are skipped."""
    _write(tmp_path / 'experiments/leaf/checkpoint', "anything")

    parametrized(config=dict(
        output_directory=str(tmp_path),
        ignore_files=['ignore',],
    ))
    # `leaf` should not be skipped, there are no non-ignored files.
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 21)


def test_parametrize_force(parametrized, tmp_path):
    """Test forcing not to skip (clears files)."""
    _write(tmp_path / 'experiments/leaf/results.txt', 123)
    _write(tmp_path / 'experiments/leaf/checkpoint', "anything")
    # Also create a checkpoint for `dir/leaf`
    _write(tmp_path / 'experiments/dir/leaf/checkpoint', "anything")

    # Run only `leaf` with force.
    parametrized("leaf", force=True, config=dict(
        output_directory=str(tmp_path),
        checkpoint_files=('checkpoint',),
    ))
    # `leaf` should not be skipped.
    _assert_content(tmp_path / 'experiments/leaf/results.txt', 21)
    # The checkpoint file should be deleted.
    assert not Path(tmp_path / 'experiments/leaf/checkpoint').exists()
    # The checkpoint for the not-run experiment should still exist.
    assert Path(tmp_path / 'experiments/dir/leaf/checkpoint').exists()


def test_cleanup(parametrized, tmp_path):
    """Test that directories are removed if there was no output."""
    # Existing directories will _not_ be removed.
    (tmp_path / 'experiments/subdir').mkdir(parents=True)
    parametrized(config=dict(output_directory=str(tmp_path)))
    assert (tmp_path / 'experiments/subdir').is_dir()
    assert not (tmp_path / 'experiments/dir/nooutput').is_dir()
