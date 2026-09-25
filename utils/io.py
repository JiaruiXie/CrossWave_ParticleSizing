"""
utils/io.py

Utility functions for reading and writing files.

Author: Jiarui Xie
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


# ============================================================
# Directory Utilities
# ============================================================

def create_directory(path):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Directory path.
    """
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def file_exists(path):
    """
    Check whether a file exists.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    bool
    """
    return Path(path).exists()


# ============================================================
# Pickle
# ============================================================

def load_pickle(filepath):
    """
    Load a pickle file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    object
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, filepath):
    """
    Save an object as a pickle file.

    Parameters
    ----------
    obj : object

    filepath : str or Path
    """
    filepath = Path(filepath)

    create_directory(filepath.parent)

    with open(filepath, "wb") as f:
        pickle.dump(
            obj,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


# ============================================================
# NumPy
# ============================================================

def save_numpy(array, filepath):
    """
    Save a NumPy array.

    For small arrays, use np.save directly.
    For large arrays, use open_memmap to avoid Windows large-file
    writing issues.

    Parameters
    ----------
    array : ndarray

    filepath : str or Path
    """

    filepath = Path(filepath)

    create_directory(filepath.parent)

    array = np.asarray(array)

    large_file_threshold = 2 * 1024 ** 3  # 2 GB

    if array.nbytes < large_file_threshold:

        np.save(
            filepath,
            array
        )

    else:

        print(
            f"Large array detected ({array.nbytes / 1024**3:.2f} GB). "
            "Saving with memory-mapped writer..."
        )

        fp = np.lib.format.open_memmap(
            filepath,
            mode="w+",
            dtype=array.dtype,
            shape=array.shape
        )

        chunk_size = 1000

        for start in range(0, array.shape[0], chunk_size):

            end = min(
                start + chunk_size,
                array.shape[0]
            )

            fp[start:end] = array[start:end]

        del fp

def load_numpy(filepath, mmap=False):
    """
    Load a NumPy array.

    Parameters
    ----------
    filepath : str or Path

    mmap : bool, default=False
        Whether to memory-map the file.

    Returns
    -------
    ndarray
    """
    filepath = Path(filepath)

    if mmap:
        return np.load(
            filepath,
            mmap_mode="r"
        )

    return np.load(filepath)


# ============================================================
# CSV
# ============================================================

def save_csv(df, filepath, index=False):
    """
    Save a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame

    filepath : str or Path

    index : bool
    """
    filepath = Path(filepath)

    create_directory(filepath.parent)

    df.to_csv(
        filepath,
        index=index
    )


def load_csv(filepath):
    """
    Load a CSV file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pandas.DataFrame
    """
    filepath = Path(filepath)

    return pd.read_csv(filepath)


# ============================================================
# JSON
# ============================================================

def save_json(dictionary, filepath, indent=4):
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    dictionary : dict

    filepath : str or Path
    """
    filepath = Path(filepath)

    create_directory(filepath.parent)

    with open(filepath, "w") as f:
        json.dump(
            dictionary,
            f,
            indent=indent
        )


def load_json(filepath):
    """
    Load a JSON file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        return json.load(f)