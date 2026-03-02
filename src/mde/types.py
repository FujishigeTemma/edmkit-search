from typing import Protocol

import numpy as np


class PredictFn(Protocol):
    """(X_train, Y_train, X_query) -> predictions"""

    __name__: str

    def __call__(
        self, X_train: np.ndarray, Y_train: np.ndarray, X_query: np.ndarray
    ) -> np.ndarray: ...


class MetricFn(Protocol):
    """(predictions, observations) -> score"""

    __name__: str

    def __call__(self, predictions: np.ndarray, observations: np.ndarray) -> float: ...


class FilterFn(Protocol):
    """(x, Y) -> accept"""

    __name__: str

    def __call__(self, x: np.ndarray, Y: np.ndarray) -> bool: ...
