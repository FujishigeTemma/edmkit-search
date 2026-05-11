from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from edmkit.search.state import States

type Energies = npt.NDArray[np.float64]
"""
Energies is a 1D array of energy values, one for each state.
"""
type Contexts = npt.NDArray[np.float64]
"""
Contexts is an opaque ndarray that can be used to store any additional information needed for the next energy computation.
"""


@dataclass(frozen=True)
class Energy:
    initial: Callable[[], npt.NDArray[np.float64]]
    """
    initial returns the initial Contexts entry to feed into the first step call.
    """
    step: Callable[
        [States, Contexts],
        tuple[Energies, Contexts],
    ]
