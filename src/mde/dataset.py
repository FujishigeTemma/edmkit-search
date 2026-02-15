from __future__ import annotations

import math
from functools import reduce
from typing import Callable, Iterator, NamedTuple

import numpy as np

Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""Type alias for a function that takes `(x, y)` and returns `(x', y')`.
Typically a closure returned by a higher-order preprocessing or data-augmentation function.
"""


def zscore_normalize(
    data: np.ndarray,
    *,
    target: str,
) -> Transform:
    """Return a `Transform` that applies z-score normalization using statistics computed from `data`.

    Parameters
    ----------
    `data` : `np.ndarray` of shape `(T, D)` or `(N, T, D)`
        Data from which mean and std are computed.
    `target` : `str`, default `"x"`
        Which element of the `(x, y)` pair to normalize.
        `"x"` normalizes input, `"y"` normalizes output, `"both"` normalizes both
        (using the same statistics — only valid when `D_x == D_y`).

    Returns
    -------
    :type: `Transform`
        `(x, y)` -> `(x', y')` where the selected target(s) are normalized.

    Examples
    --------
    ```python
    zscore_x = zscore_normalize(X_train, target="x")        # normalize input
    zscore_y = zscore_normalize(Y_train, target="y")        # normalize output
    tf = compose(zscore_x, zscore_y)                        # both, independent stats
    ```
    """
    if target not in ("x", "y", "both"):
        raise ValueError(f"target must be 'x', 'y', or 'both', got {target!r}")

    flat = data.reshape(-1, data.shape[-1])  # (N*T, D) or (T, D)
    mean = flat.mean(axis=0).astype(np.float32)  # (D,)
    std = np.clip(flat.std(axis=0).astype(np.float32), 1e-8, None)  # (D,)

    def normalize(a: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    if target == "x":

        def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (normalize(x), y)
    elif target == "y":

        def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (x, normalize(y))
    else:

        def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (normalize(x), normalize(y))

    return transform


def gaussian_noise(
    sigma: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Transform:
    """Return a `Transform` that adds Gaussian noise to the input (for data augmentation).

    Parameters
    ----------
    `sigma` : `float`, default `0.1`
        Standard deviation of the noise.
    `rng` : `np.random.Generator` or `None`, default `None`
        Random number generator for reproducibility.
        If `None`, a new unseeded generator is created.

    Returns
    -------
    :type: `Transform`
        `(x, y)` -> `(x + noise, y)`
    """
    rng = np.random.default_rng(rng)

    def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise = rng.standard_normal(x.shape).astype(x.dtype) * sigma
        return (x + noise, y)

    return transform


def compose(*transforms: Transform) -> Transform:
    """Return a `Transform` that applies multiple transforms in left-to-right order.

    Examples
    --------
    ```python
    transform = compose(zscore_normalize(X_train), gaussian_noise(0.05))
    # zscore is applied first, then noise
    ```
    """

    def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return reduce(lambda xy, fn: fn(*xy), transforms, (x, y))

    return transform


class Dataset:
    """Time series dataset for X -> Y mapping.

    Parameters
    ----------
    `X` : `np.ndarray` of shape `(T, D_x)`
        Input time series.
    `Y` : `np.ndarray` of shape `(T, D_y)` or `(T,)`
        Output time series. 1D is auto-promoted to `(T, 1)`.
    `transform` : :type: `Transform` or `None`, default `None`
        `(x, y)` -> `(x', y')` closure for preprocessing / augmentation.

    Raises
    ------
    `ValueError`
        If `X` is not 2-dimensional, `Y` is not 1D or 2D,
        or if `T` dimensions mismatch.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        transform: Transform | None = None,
    ):
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional (T, D_x), got shape {X.shape}"
            )
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if Y.ndim != 2:
            raise ValueError(
                f"Y must be 1D or 2D, got shape {Y.shape}"
            )
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"T mismatch: X has {X.shape[0]} timesteps, Y has {Y.shape[0]}"
            )

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].copy()
        y = self.Y[idx].copy()
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y


class Subset:
    """A view into a `Dataset` selected by index, without copying data.

    Parameters
    ----------
    `dataset` : :type: `Dataset`
        The underlying dataset.
    `indices` : `np.ndarray`
        Indices into `dataset` to expose.
    """

    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]


def temporal_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    *,
    gap: int = 0,
) -> tuple[Subset, Subset]:
    """Split a dataset temporally into train and validation subsets.

    Layout::

        [===== Train =====][gap][== Val ==]

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    train_ratio : float
        Fraction of data for training.
    gap : int
        Number of points to skip between splits (prevents leakage).

    Returns
    -------
    tuple[Subset, Subset]
        (train_subset, val_subset)

    Raises
    ------
    ValueError
        If ratio is invalid or any split would be empty.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    n = len(dataset)
    train_end = int(n * train_ratio)
    val_start = train_end + gap

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(val_start, n)

    if len(train_indices) == 0:
        raise ValueError("Training set is empty")
    if len(val_indices) == 0:
        raise ValueError("Validation set is empty")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
    )


class DataLoader:
    """Mini-batch iterator over a `Dataset`.

    Parameters
    ----------
    `dataset` : :type: `Dataset`
        Dataset instance to iterate over.
    `batch_size` : `int`, default `32`
        Mini-batch size.
    `shuffle` : `bool`, default `False`
        Whether to shuffle indices each epoch.
    `drop_last` : `bool`, default `False`
        Whether to drop the last incomplete batch.
    `collate_fn` : `Callable` or `None`, default `None`
        Batch collation function. Defaults to `np.stack` along a new axis.
    `rng` : `np.random.Generator` or `None`, default `None`
        Random number generator for reproducible shuffling.
        If `None`, a new unseeded generator is created.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self._default_collate
        self._rng = np.random.default_rng(rng)

    # ---- Default collate: list[tuple] -> tuple[np.ndarray, ...] ----
    @staticmethod
    def _default_collate(
        batch: list[tuple[np.ndarray, ...]],
    ) -> tuple[np.ndarray, ...]:
        """Stack arrays at each position across samples.

        Example: `[(x0, y0), (x1, y1), ...]` -> `(np.stack([x0, x1, ...]), np.stack([y0, y1, ...]))`
        """
        return tuple(np.stack(arrays) for arrays in zip(*batch))

    def __iter__(self) -> Iterator[tuple[np.ndarray, ...]]:
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start : min(end, n)]
            batch = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)


class Fold(NamedTuple):
    """A single train/validation split for time-series cross-validation."""

    train: np.ndarray
    val: np.ndarray


def make_expanding_windows(
    n: int,
    *,
    min_train: int,
    val_size: int,
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
    `val_size` : `int`
        Number of validation samples per fold.
    `gap` : `int`
        Number of points to skip between train and val (prevents leakage).

    Returns
    -------
    `list[Fold]`
        List of folds.

    Examples
    --------
    ```python
    folds = make_expanding_windows(20, min_train=10, val_size=5)
    # fold 0: train=[0..9]   val=[10..14]
    # fold 1: train=[0..14]  val=[15..19]
    ```
    """
    folds: list[Fold] = []
    val_start = min_train + gap
    while val_start + val_size <= n:
        train_end = val_start - gap
        train_idx = np.arange(train_end)
        val_idx = np.arange(val_start, val_start + val_size)
        folds.append(Fold(train_idx, val_idx))
        val_start += val_size
    return folds


def make_sliding_windows(
    n: int,
    *,
    train_size: int,
    val_size: int,
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
    `val_size` : `int`
        Number of validation samples per fold.
    `gap` : `int`
        Number of points to skip between train and val (prevents leakage).

    Returns
    -------
    `list[Fold]`
        List of folds.

    Examples
    --------
    ```python
    folds = make_sliding_windows(20, train_size=10, val_size=5)
    # fold 0: train=[0..9]   val=[10..14]
    # fold 1: train=[5..14]  val=[15..19]
    ```
    """
    folds: list[Fold] = []
    val_start = train_size + gap
    while val_start + val_size <= n:
        train_start = val_start - gap - train_size
        train_idx = np.arange(train_start, train_start + train_size)
        val_idx = np.arange(val_start, val_start + val_size)
        folds.append(Fold(train_idx, val_idx))
        val_start += val_size
    return folds
