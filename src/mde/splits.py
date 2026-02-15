"""Temporal splitting for time series data."""

from typing import NamedTuple

import numpy as np


class Split(NamedTuple):
    """Three-way temporal split of time series data.

    Attributes
    ----------
    train : np.ndarray
        Training indices.
    val : np.ndarray
        Validation indices.
    test : np.ndarray
        Test indices.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def temporal_split(
    length: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    *,
    gap: int = 0,
) -> Split:
    """Split time series indices temporally.

    Layout::

        [===== Train =====][gap][== Val ==][gap][== Test ==]

    Parameters
    ----------
    length : int
        Total length of the time series.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.
    gap : int
        Number of points to skip between splits (prevents leakage).

    Returns
    -------
    Split
        Split indices.

    Raises
    ------
    ValueError
        If ratios are invalid or any split would be empty.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if not train_ratio + val_ratio < 1:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}"
        )
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))

    train = np.arange(0, train_end)
    val = np.arange(train_end + gap, val_end)
    test = np.arange(val_end + gap, length)

    if len(train) == 0:
        raise ValueError("Training set is empty")
    if len(val) == 0:
        raise ValueError("Validation set is empty")
    if len(test) == 0:
        raise ValueError("Test set is empty")

    return Split(
        train=train,
        val=val,
        test=test,
    )
