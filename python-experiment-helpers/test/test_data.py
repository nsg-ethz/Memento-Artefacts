"""Test data utilities."""

import experiment_helpers as eh
import numpy as np
import pandas
import pytest


def test_gzip_write(tmp_path):
    """Test gzip compression."""
    _file = tmp_path / "test"
    _file_uncompressed = tmp_path / "test.big"
    data = bytes([42] * 100)

    with eh.data.open(_file, 'wb', compression="gzip") as file:
        file.write(data)
    with eh.data.open(_file_uncompressed, 'wb') as file:
        file.write(data)

    # Compare uncompressed lengths.
    with open(_file, 'rb') as file_a, open(_file_uncompressed, 'rb') as file_b:
        assert len(file_a.read()) < len(file_b.read())


def test_auto_gzip(tmp_path):
    """Test auto compression based on suffix."""
    _filegzip = tmp_path / f"test.gz"
    data = [42] * 100

    eh.data.to_pickle(data, _filegzip)
    # Infer decompression.
    assert data == eh.data.read_pickle(_filegzip)
    # Explicit decompression.
    assert data == eh.data.read_pickle(_filegzip,
                                       open_kwargs={'compression': 'gzip'})


@pytest.mark.parametrize("suffix", ['', '.gz'])
def test_pickle(tmp_path, suffix):
    """Test pickling and unpickling data."""
    _file = tmp_path / f"test{suffix}"
    data = {"some": "data"}
    eh.data.to_pickle(data, _file)
    assert data == eh.data.read_pickle(_file)


def test_gzip_pickle(tmp_path):
    """Test gzip compression."""
    _file_explicit = tmp_path / "test"
    _file_inferred = tmp_path / "test.gz"
    _file_uncompressed = tmp_path / "test.big"
    data = [42] * 100

    eh.data.to_pickle(data, _file_explicit,
                      open_kwargs={'compression': 'gzip'})
    eh.data.to_pickle(data, _file_inferred)
    eh.data.to_pickle(data, _file_uncompressed)

    with open(_file_explicit, 'rb') as file_explicit, \
            open(_file_inferred, 'rb') as file_inferred, \
            open(_file_uncompressed, 'rb') as file_uncompressed:
        len_explicit = len(file_explicit.read())
        len_inferred = len(file_inferred.read())
        len_uncompressed = len(file_uncompressed.read())

        assert len_uncompressed > len_explicit
        assert len_explicit == len_inferred


@pytest.mark.parametrize("extension", ['.npz', '.npz.gz'])
def test_npz(tmp_path, extension):
    """Test saving and loading numpy arrays.

    Extensions with compression are ignored, but cause no issues.
    """
    data = {
        "a": np.arange(10),
        "b": np.zeros(100),
    }
    filename = tmp_path / f"test{extension}"
    eh.data.to_npz(data, filename)
    loaded = eh.data.read_npz(filename)
    for key, expected in data.items():
        np.testing.assert_equal(loaded[key], expected)


def test_npz_compression_forbidded(tmp_path):
    """Forcing a compression for npz is forbidden."""
    data = {
        "a": np.arange(10),
        "b": np.zeros(100),
    }
    filename = tmp_path / "test.npz.gz"
    with pytest.raises(AssertionError):
        eh.data.to_npz(data, filename, open_kwargs={'compression': 'gzip'})
    with pytest.raises(AssertionError):
        eh.data.read_npz(filename, open_kwargs={'compression': 'gzip'})


@pytest.mark.parametrize("extension", ['.csv', '.csv.gz'])
def test_csv(tmp_path, extension):
    """Testing reading and writing a DataFrame."""
    data = pandas.DataFrame({
        "a": np.arange(10),
        "b": np.zeros(10),
    })
    filename = tmp_path / f"test{extension}"
    assert not filename.exists()
    eh.data.to_csv(data, filename)
    loaded = eh.data.read_csv(filename)
    pandas.testing.assert_frame_equal(data, loaded)


@pytest.mark.parametrize("obj", [{"a": 1}, [1, 2, 3], 42, "string"])
@pytest.mark.parametrize("extension", ['.json', '.json.gz'])
def test_json(tmp_path, obj, extension):
    """Test writing/reading json."""
    filename = tmp_path / f"test{extension}"
    assert not filename.exists()
    eh.data.to_json(obj, filename)
    loaded = eh.data.read_json(filename)
    assert obj == loaded
