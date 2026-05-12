from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

from edmkit.search.state import States

type Energies = npt.NDArray[np.float64]
"""
Energies is a 1D array of energy values, one per state.
"""

type Contexts = npt.NDArray[np.float64]
"""
Contexts is an opaque ndarray that can be used to store any additional information needed for the next energy computation.
"""

type Plan = Callable[
    [States, Contexts],
    Iterable[Callable[[], tuple[slice, Energies, Contexts]]],
]

type Energy = Callable[[States, Contexts], tuple[Energies, Contexts]]
