"""Test fixtures."""
# pylint: disable=redefined-outer-name,import-outside-toplevel

from pathlib import Path
from typing import Callable, Union

import pytest

TESTDATADIR = Path(__file__).parent.joinpath('testdata')


@pytest.fixture
def testdatapath() -> Callable[..., Path]:
    """Fixture that returns paths to testdata."""
    def _get_path(*filenames: Union[str, Path]) -> Path:
        filepath = Path(*filenames)
        _path = TESTDATADIR.joinpath(filepath)
        assert _path.exists(), f"There is no test data `{filepath}`."
        return _path

    return _get_path
