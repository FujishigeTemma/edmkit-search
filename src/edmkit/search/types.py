from typing import NamedTuple, Protocol

import numpy as np


class FilterFn(Protocol):
    """(x, Y) -> accept"""

    __name__: str

    def __call__(self, x: np.ndarray, Y: np.ndarray) -> bool: ...


class SampleLossFn(Protocol):
    """(predictions, observations) -> per-sample lower-is-better objective."""

    __name__: str

    def __call__(
        self, predictions: np.ndarray, observations: np.ndarray
    ) -> np.ndarray: ...


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
