from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type WeightFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def normalized(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have a positive finite sum")
    return values / total


def softmax(temperature: float = 1.0) -> WeightFunc:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def weight(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        state = np.asarray(values, dtype=np.float64)
        if state.ndim != 1:
            raise ValueError(f"weight state must be 1D, got {state.ndim}D")
        if state.size == 0:
            return np.empty(0, dtype=np.float64)
        logits = state / temperature
        logits = logits - np.max(logits)
        return normalized(np.exp(logits))

    return weight
