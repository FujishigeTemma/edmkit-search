import numpy as np
import numpy.typing as npt

type States = npt.NDArray[np.int64]
"""A batch of search states of shape ``(N, d)``: ``N`` states, each holding ``d`` indices into the original dataset. Within a single step, all states in the batch share the same length ``d``."""


def initial() -> States:
    """Build the empty initial batch — a single zero-length state.

    Returns
    -------
    States
        Array of shape ``(1, 0)``.
    """
    return np.empty((1, 0), dtype=np.int64)
