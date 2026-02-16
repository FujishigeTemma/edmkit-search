from collections.abc import Callable
from functools import reduce
from typing import TypeAlias

import numpy as np

Transform: TypeAlias = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
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
