"""Data splitting functions for time series validation."""

import numpy as np

from .types import DataSplit


def split_data(
    length: int,
    train_ratio: float,
    val_ratio: float,
    *,
    gap: int = 0,
) -> DataSplit:
    """Split time series indices into train/validation/test sets with optional gap.

    Parameters
    ----------
    length : int
        Total length of the time series.
    train_ratio : float
        Ratio of data for training (e.g., 0.6).
    val_ratio : float
        Ratio of data for validation (e.g., 0.2).
    gap : int, optional
        Number of points to skip between splits to prevent temporal leakage.
        Default is 0.

    Returns
    -------
    DataSplit
        Dataclass containing indices for train, val, test sets.

    Raises
    ------
    ValueError
        If ratios are invalid or if any split would be empty.

    Notes
    -----
    Data layout with gap::

        [===== Train =====][gap][== Val ==][gap][== Test ==]
             (60%)                (20%)            (20%)
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

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end + gap, val_end)
    test_indices = np.arange(val_end + gap, length)

    if len(train_indices) == 0:
        raise ValueError("Training set is empty")
    if len(val_indices) == 0:
        raise ValueError("Validation set is empty")
    if len(test_indices) == 0:
        raise ValueError("Test set is empty")

    return DataSplit(
        train_end=train_end,
        val_end=val_end,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
