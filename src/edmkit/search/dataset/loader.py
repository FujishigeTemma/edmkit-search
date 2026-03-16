import math
from collections.abc import Callable, Iterator

import numpy as np

from .containers import Dataset


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
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
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
