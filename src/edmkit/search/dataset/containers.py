from functools import cached_property

import numpy as np

from .transforms import Transform


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
            raise ValueError(f"X must be 2-dimensional (T, D_x), got shape {X.shape}")
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if Y.ndim != 2:
            raise ValueError(f"Y must be 1D or 2D, got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"T mismatch: X has {X.shape[0]} timesteps, Y has {Y.shape[0]}")

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


class Subset(Dataset):
    """A view into a `Dataset` selected by index, without copying data.

    This is a `Dataset` subtype, so row views can be passed anywhere a
    dataset is expected.

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

    @cached_property
    def X(self) -> np.ndarray:
        return self.dataset.X[self.indices]

    @cached_property
    def Y(self) -> np.ndarray:
        return self.dataset.Y[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]
