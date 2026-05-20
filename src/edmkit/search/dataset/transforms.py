from collections.abc import Callable
from functools import reduce

import numpy as np

type Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""A function ``(x, y) -> (x', y')`` applied lazily at sample-access time. Typically a closure returned by a preprocessing or data-augmentation factory."""


def zscore_normalize(
    data: np.ndarray,
    *,
    target: str,
) -> Transform:
    """Build a `Transform` that z-score normalizes using statistics from ``data``.

    The mean and standard deviation are computed once over the leading
    axes of ``data`` (treating the last axis as the feature axis) and
    captured in the returned closure. The closure can then be applied
    repeatedly to individual samples without recomputing statistics.

    Parameters
    ----------
    data : np.ndarray of shape (T, D) or (N, T, D)
        Reference data from which the per-feature mean and standard
        deviation are computed. The last axis is treated as the
        feature axis.
    target : {"x", "y", "both"}
        Which arm of the ``(x, y)`` pair to normalize. ``"both"`` is
        only valid when ``D_x == D_y`` (the same statistics are used
        for both arms).

    Returns
    -------
    Transform
        ``(x, y) -> (x', y')`` where the selected arm(s) are
        normalized.

    Raises
    ------
    ValueError
        If ``target`` is not one of ``"x"``, ``"y"``, ``"both"``.

    Examples
    --------
    ```python
    zscore_x = zscore_normalize(X_train, target="x")
    zscore_y = zscore_normalize(Y_train, target="y")
    transform = compose(zscore_x, zscore_y)  # independent stats per arm
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
    """Build a `Transform` that perturbs the input with Gaussian noise.

    The noise is drawn at sample-access time, so each pass through
    the dataset sees fresh noise — making this suitable for
    on-the-fly data augmentation. Only the input arm ``x`` is
    perturbed; ``y`` is returned unchanged.

    Parameters
    ----------
    sigma : float, default 0.1
        Standard deviation of the noise.
    rng : np.random.Generator or None, default None
        Random number generator for reproducibility. When ``None``,
        a fresh unseeded generator is created.

    Returns
    -------
    Transform
        ``(x, y) -> (x + noise, y)``.
    """
    rng = np.random.default_rng(rng)

    def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise = rng.standard_normal(x.shape).astype(x.dtype) * sigma
        return (x + noise, y)

    return transform


def compose(*transforms: Transform) -> Transform:
    """Compose multiple transforms into a single left-to-right pipeline.

    Parameters
    ----------
    *transforms : Transform
        Transforms to apply in order. The output of each transform
        feeds the next.

    Returns
    -------
    Transform
        ``(x, y) -> transforms[-1](... transforms[1](transforms[0](x, y)))``.

    Examples
    --------
    ```python
    transform = compose(zscore_normalize(X_train, target="x"), gaussian_noise(0.05))
    # z-score is applied first, then noise is added on top
    ```
    """

    def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return reduce(lambda xy, fn: fn(*xy), transforms, (x, y))

    return transform
