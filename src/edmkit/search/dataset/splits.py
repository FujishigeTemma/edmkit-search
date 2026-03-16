from typing import NamedTuple

import numpy as np


class Fold(NamedTuple):
    """A single train/validation split for time-series cross-validation."""

    train: np.ndarray
    validation: np.ndarray


def temporal_split(
    n: int,
    train_ratio: float = 0.8,
    *,
    gap: int = 0,
) -> Fold:
    """Split a time series temporally into train and validation index arrays.

    Layout::

        [===== Train =====][gap][== Val ==]

    Parameters
    ----------
    n : int
        Total number of samples.
    train_ratio : float
        Fraction of data for training.
    gap : int
        Number of points to skip between splits (prevents leakage).

    Returns
    -------
    Fold
        A single fold with train and validation indices.

    Raises
    ------
    ValueError
        If ratio is invalid or any split would be empty.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    train_end = int(n * train_ratio)
    validation_start = train_end + gap

    train_indices = np.arange(0, train_end)
    validation_indices = np.arange(validation_start, n)

    if len(train_indices) == 0:
        raise ValueError("Training set is empty")
    if len(validation_indices) == 0:
        raise ValueError("Validation set is empty")

    return Fold(train_indices, validation_indices)


def expanding_splits(
    n: int,
    *,
    min_train: int,
    validation_size: int,
    gap: int = 0,
) -> list[Fold]:
    """Generate `Fold`s for expanding-window time-series cross-validation.

    The training set grows with each fold while the validation window slides forward.

    Parameters
    ----------
    `n` : `int`
        Total number of samples.
    `min_train` : `int`
        Minimum number of training samples (used for the first fold).
    `validation_size` : `int`
        Number of validation samples per fold.
    `gap` : `int`
        Number of points to skip between train and validation (prevents leakage).

    Returns
    -------
    `list[Fold]`
        List of folds.

    Raises
    ------
    `ValueError`
        If `n`, `min_train`, or `validation_size` are not positive, or `gap` is negative.

    Examples
    --------
    ```python
    folds = expanding_splits(20, min_train=10, validation_size=5)
    # fold 0: train=[0..9]   validation=[10..14]
    # fold 1: train=[0..14]  validation=[15..19]
    ```
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if min_train <= 0:
        raise ValueError(f"min_train must be positive, got {min_train}")
    if validation_size <= 0:
        raise ValueError(f"validation_size must be positive, got {validation_size}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    folds: list[Fold] = []
    validation_start = min_train + gap
    while validation_start + validation_size <= n:
        train_end = validation_start - gap
        train_idx = np.arange(train_end)
        validation_idx = np.arange(validation_start, validation_start + validation_size)
        folds.append(Fold(train_idx, validation_idx))
        validation_start += validation_size
    return folds


def sliding_splits(
    n: int,
    *,
    train_size: int,
    validation_size: int,
    gap: int = 0,
) -> list[Fold]:
    """Generate `Fold`s for sliding-window time-series cross-validation.

    A fixed-size training window slides forward together with the validation window.

    Parameters
    ----------
    `n` : `int`
        Total number of samples.
    `train_size` : `int`
        Fixed number of training samples per fold.
    `validation_size` : `int`
        Number of validation samples per fold.
    `gap` : `int`
        Number of points to skip between train and validation (prevents leakage).

    Returns
    -------
    `list[Fold]`
        List of folds.

    Raises
    ------
    `ValueError`
        If `n`, `train_size`, or `validation_size` are not positive, or `gap` is negative.

    Examples
    --------
    ```python
    folds = sliding_splits(20, train_size=10, validation_size=5)
    # fold 0: train=[0..9]   validation=[10..14]
    # fold 1: train=[5..14]  validation=[15..19]
    ```
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if validation_size <= 0:
        raise ValueError(f"validation_size must be positive, got {validation_size}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    folds: list[Fold] = []
    validation_start = train_size + gap
    while validation_start + validation_size <= n:
        train_start = validation_start - gap - train_size
        train_idx = np.arange(train_start, train_start + train_size)
        validation_idx = np.arange(validation_start, validation_start + validation_size)
        folds.append(Fold(train_idx, validation_idx))
        validation_start += validation_size
    return folds
