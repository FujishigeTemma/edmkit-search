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
    `data` : `np.ndarray` of shape `(N, T, D)`
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

    flat = data.reshape(-1, data.shape[-1])  # (N*T, D)
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
    """Dataset for time-series X -> Y mapping.

    Parameters
    ----------
    `X` : `np.ndarray` of shape `(N, T, D_x)`
        Input time series.
    `Y` : `np.ndarray` of shape `(N, T, D_y)`
        Output time series.
    `window` : `int` or `None`, default `None`
        Sub-sequence length to extract. If `None`, the full length `T` is used.
    `stride` : `int` or `None`, default `None`
        Step size between consecutive windows. Defaults to `window`.
    `transform` : :type: `Transform` or `None`, default `None`
        `(x, y)` -> `(x', y')` closure for preprocessing / augmentation.

    Raises
    ------
    `ValueError`
        If `X` or `Y` are not 3-dimensional, have mismatched `N` or `T`,
        or if `window` exceeds `T`.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        window: int | None = None,
        stride: int | None = None,
        transform: Transform | None = None,
    ):
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3-dimensional (N, T, D_x), got shape {X.shape}"
            )
        if Y.ndim != 3:
            raise ValueError(
                f"Y must be 3-dimensional (N, T, D_y), got shape {Y.shape}"
            )
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"N mismatch: X has {X.shape[0]} trials, Y has {Y.shape[0]}"
            )
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"T mismatch: X has {X.shape[1]} timesteps, Y has {Y.shape[1]}"
            )

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.transform = transform

        N, T, _ = self.X.shape
        self.window = window or T
        self.stride = stride or self.window

        if self.window > T:
            raise ValueError(f"window ({self.window}) exceeds T ({T})")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        # ---- Window -> (trial_idx, start_idx) mapping ----
        self._index_map: list[tuple[int, int]] = []
        for trial in range(N):
            for start in range(0, T - self.window + 1, self.stride):
                self._index_map.append((trial, start))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        trial, start = self._index_map[idx]
        end = start + self.window
        x = self.X[trial, start:end].copy()
        y = self.Y[trial, start:end].copy()
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

    Examples
    --------
    ```python
    ds = Dataset(X, Y, window=100, stride=50)

    folds = make_expanding_windows(len(ds), min_train=100, val_size=20)
    for fold in folds:
        train_ds = Subset(ds, fold.train)   # no data copy
        val_ds   = Subset(ds, fold.val)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=32)
    ```
    """

    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]


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
    val_start = min_train
    while val_start + val_size <= n:
        train_idx = np.arange(val_start)
        val_idx = np.arange(val_start, val_start + val_size)
        folds.append(Fold(train_idx, val_idx))
        val_start += val_size
    return folds


def make_sliding_windows(
    n: int,
    *,
    train_size: int,
    val_size: int,
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
    val_start = train_size
    while val_start + val_size <= n:
        train_idx = np.arange(val_start - train_size, val_start)
        val_idx = np.arange(val_start, val_start + val_size)
        folds.append(Fold(train_idx, val_idx))
        val_start += val_size
    return folds
