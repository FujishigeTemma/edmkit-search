from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type WeightFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
"""A weighting function ``(N, K) -> (N, K)`` that turns a context matrix into per-row weights. Rows are expected to sum to 1 so the result behaves as an attention distribution over the K columns."""


def softmax(temperature: float = 1.0) -> WeightFunc:
    """Build a row-wise softmax `WeightFunc` with the given temperature.

    Lower temperature concentrates weight on the columns with the
    largest values; higher temperature flattens toward a uniform
    distribution. The implementation subtracts the per-row maximum
    before exponentiating, so it is numerically stable for any input
    scale.

    Parameters
    ----------
    temperature : float, default 1.0
        Positive scalar controlling the sharpness of the distribution.
        Must be strictly positive.

    Returns
    -------
    WeightFunc
        ``(N, K) -> (N, K)`` row-stochastic weighting.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def weight(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        logits = values / temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        return exps / exps.sum(axis=1, keepdims=True)

    return weight
