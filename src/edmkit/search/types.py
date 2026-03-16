from typing import NamedTuple, Protocol

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


class Step(NamedTuple):
    """Result of a single variable selection step.

    Parameters
    ----------
    index : int
        Variable selected at this step.
    score : float
        Score of the current selection.
    selected : tuple[int, ...]
        All selected variable indices (full state).
    """

    index: int
    score: float
    selected: tuple[int, ...]


class Selection(NamedTuple):
    """Result of the selection phase.

    Parameters
    ----------
    indices : list[int]
        Indices of selected variables.
    scores : list[float]
        Score at each dimension step.
    """

    indices: list[int]
    scores: list[float]
