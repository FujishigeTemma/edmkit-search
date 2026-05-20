from functools import cached_property

import numpy as np

from .transforms import Transform


class Dataset:
    """Time-series dataset of paired input/output sequences.

    Holds an input array ``X`` and a target array ``Y`` sharing a common
    time axis, plus an optional `Transform` applied lazily at
    ``__getitem__`` time. Inputs are cast to ``float32`` and a 1D ``Y``
    is auto-promoted to ``(T, 1)`` so downstream code can treat the
    target as 2D uniformly.

    Parameters
    ----------
    X : np.ndarray of shape (T, D_x)
        Input time series.
    Y : np.ndarray of shape (T, D_y) or (T,)
        Target time series. 1D input is promoted to ``(T, 1)``.
    transform : Transform or None, default None
        ``(x, y) -> (x', y')`` closure applied at indexing time. Used
        for preprocessing or on-the-fly data augmentation.

    Raises
    ------
    ValueError
        - If ``X`` is not 2-dimensional.
        - If ``Y`` is not 1D or 2D.
        - If ``X`` and ``Y`` have different lengths along the time axis.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.search.dataset import Dataset, zscore_normalize

    X = np.random.default_rng(0).standard_normal((1000, 8))
    Y = X[:, :1]

    data = Dataset(X, Y, transform=zscore_normalize(X, target="x"))
    x, y = data[0]
    ```
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
    """A non-copying view into a `Dataset` restricted to selected rows.

    ``Subset`` is a ``Dataset`` subtype, so it can be passed anywhere
    a dataset is expected (e.g. as the train/validation arms of a
    fold). ``X`` and ``Y`` are materialized lazily via
    ``cached_property``.

    Parameters
    ----------
    dataset : Dataset
        Underlying dataset.
    indices : np.ndarray
        1D integer indices selecting rows of ``dataset`` to expose.
        The exposed length is ``len(indices)``.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.search.dataset import Dataset, Subset

    data = Dataset(X, Y)
    train = Subset(data, np.arange(800))
    validation = Subset(data, np.arange(800, 1000))
    ```
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
