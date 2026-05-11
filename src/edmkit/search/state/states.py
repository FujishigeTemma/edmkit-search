import numpy as np
import numpy.typing as npt

type States = npt.NDArray[np.int64]
"""
States is a 2D array of shape (N, d): N states, each holding d indices into the original dataset.
A single step's batch always contains states of equal length d.
"""


def initial() -> States:
    """Initial states batch: a single empty state, shape (1, 0)."""
    return np.empty((1, 0), dtype=np.int64)
