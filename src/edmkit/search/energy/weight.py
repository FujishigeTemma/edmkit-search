from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type WeightFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def softmax(temperature: float = 1.0) -> WeightFunc:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def weight(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        logits = values / temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        return exps / exps.sum(axis=1, keepdims=True)

    return weight
