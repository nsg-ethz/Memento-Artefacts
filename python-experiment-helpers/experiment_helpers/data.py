"""Data loading and storing helpers.

Offers one-liners to save and load common datatypes with some additional bells
and whistles. The functions are based on fsspec and thus support both
compression (inferred from extension automatically) and remote file systems.

E.g. open('ssh://host/path.gz') opens a remote file using gzip.

As in particular compression can take a while and leave unuseable files if
interrupted, all writing is first done to a temporary file and only renamed
to the final name if everything went well.

Concrete functions:

- `to_json` and `read_json` for json files. Good compression with gzip while
    being easy to use. Recommended! E.g., `to_json(data, 'data.json.gz')`.
- `to_csv` and `read_csv` for pandas DataFrames. Good compression with gzip
    while being easy to use. Recommended! E.g., `to_csv(df, 'data.csv.gz')`.
- `to_npz` for dictionaries with numpy arrays. npz has it's own compression
    and ignores all provided compression options or extensions.
- `to_pickle` and `read_pickle` for pickling arbitrary data.
    Not recommended for long-term storage or portability.
"""
# pylint: disable=redefined-builtin, dangerous-default-value
# We only use dicts as defaults that are read, we never write to them.

import json
import os
import pickle
import uuid
import warnings
from contextlib import ExitStack, contextmanager
from functools import wraps
from tempfile import NamedTemporaryFile
from typing import Dict, List, Union, Any

import fsspec
import fsspec.asyn
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from upath import UPath as Path

PathType = Union[os.PathLike, str]
JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]


# Make async fsspec filesystems work with fork. Concrete example: gcsfs.
# https://github.com/fsspec/gcsfs/issues/379
os.register_at_fork(after_in_child=fsspec.asyn.reset_lock)


def open(urlpath, mode: str = 'r', compression: str = "infer", **kwargs):
    """Open a file using fsspec and infer compression from extension."""
    return fsspec.open(urlpath, mode=mode, compression=compression, **kwargs)


# Main usecase is to use univeral-pathlib with sshfs, which is not yet
# officially supported. It works though, so we can ignore the warning.
warnings.filterwarnings(
    "ignore", message="UPath 'ssh' filesystem not explicitly implemented.")


def to_pickle(data, path: PathType, open_kwargs={}, **kwargs):
    """Pickle arbitrary data."""
    with write_tmp(path, 'wb', **open_kwargs) as file:
        pickle.dump(data, file, **kwargs)


def read_pickle(path: PathType, open_kwargs={}, **kwargs):
    """Helper to unpickle data."""
    with open(path, 'rb', **open_kwargs) as file:
        return pickle.load(file, **kwargs)


def to_json(data: JsonType, path: PathType, open_kwargs={}, **kwargs):
    """Save data as json."""
    with write_tmp(path, 'w', **open_kwargs) as file:
        json.dump(data, file, **kwargs)


def read_json(path: PathType, open_kwargs={}, **kwargs) -> JsonType:
    """Load json data."""
    with open(path, 'r', **open_kwargs) as file:
        return json.load(file, **kwargs)


def to_npz(data: Dict[str, ArrayLike], path: PathType, open_kwargs={}):
    """Store dictionaries of numpy arrays. Always compressed.

    Does not support arbitrary compression specified by user or extension.
    """
    open_kwargs = open_kwargs.copy()
    open_kwargs.setdefault('compression', None)
    assert open_kwargs['compression'] is None, \
        "npz must write to uncompressed files."
    with write_tmp(path, 'wb', **open_kwargs) as file:
        np.savez_compressed(file, **data)


def read_npz(path: Union[str, os.PathLike], open_kwargs={}, **kwargs) \
        -> Dict[str, ArrayLike]:
    """Load .npz files and return dict."""
    open_kwargs = open_kwargs.copy()
    open_kwargs.setdefault('compression', None)
    assert open_kwargs['compression'] is None, \
        "npz must read from uncompressed files."
    with open(path, 'rb', **open_kwargs) as file:
        return dict(np.load(file, **kwargs))


def to_np(data: ArrayLike, path: PathType, open_kwargs={}, **kwargs):
    """Save numpy arrays."""
    with write_tmp(path, 'wb', **open_kwargs) as file:
        np.save(file, data, **kwargs)


def read_np(path: Union[str, os.PathLike], open_kwargs={}, **kwargs) \
        -> ArrayLike:
    """Load .npz files and return dict."""
    with open(path, 'rb', **open_kwargs) as file:
        return np.load(file, **kwargs)


def to_csv(data: pd.DataFrame, path: PathType, open_kwargs={},
           index=False, **kwargs):
    """Write DataFrames to csv. Using compression is recommended!

    Simply provide a file ending such as `.gz`.
    Opposed to pandas `to_csv`, this method does not save the index by default,
    but you can pass `index=True` to change that.
    """
    with write_tmp(path, 'wb', **open_kwargs) as file:
        data.to_csv(file, index=index, **kwargs)


def read_csv(path: PathType, open_kwargs={}, **kwargs) -> pd.DataFrame:
    """Read DataFrames from csv."""
    with open(path, 'rb', **open_kwargs) as file:
        return pd.read_csv(file, **kwargs)


@contextmanager
def write_tmp(path: PathType, *args, **kwargs):
    """Open a temprary file for writing, rename to path after closing."""
    path = Path(path).resolve()
    extensions = "".join(path.suffixes)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    # Find an unused temprary file name with the same extension
    # (so that compression is inferred correctly).
    while True:
        tmppath = directory / f".{uuid.uuid4().hex}{extensions}"
        if not tmppath.exists():
            break
    try:
        with open(tmppath, *args, **kwargs) as file:
            yield file
        # Rename to final name after file is closed.
        tmppath.rename(path)
    finally:
        tmppath.unlink(missing_ok=True)


@contextmanager
def temppickle(*items):
    """Pickle all items and yield file names. Use as contextmanager."""
    with ExitStack() as stack:
        filenames = []
        for item in items:
            file = stack.enter_context(NamedTemporaryFile())
            to_pickle(item, file)
            file.seek(0)  # Reopening and parsing fails otherwise for reasons
            filenames.append(file.name)
        if len(filenames) == 1:
            yield filenames[0]
        else:
            yield filenames


def unpickle_args(n_args: int):
    """Load the first n arguments using pickle.

    Arguments are only loaded if they are strings (maybe check if they are
    paths?) to allow passing values directly for debugging.
    """
    def _load_n(func):
        @wraps(func)
        def _load(*args, **kwargs):
            unpickled = [read_pickle(arg) if isinstance(arg, str) else arg
                         for arg in args[:n_args]]
            return func(*unpickled, *args[n_args:], **kwargs)
        return _load
    return _load_n
