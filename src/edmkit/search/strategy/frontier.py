from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edmkit.search.energy import Contexts, Energies
from edmkit.search.state import States


@dataclass(frozen=True)
class Frontier:
    """Immutable batch of search candidates carried between steps.

    The three arrays are aligned along their leading axis: row ``i`` of
    ``states`` corresponds to ``contexts[i]`` and ``energies[i]``. The
    frontier is what each ``Step`` consumes and produces; ``run`` then
    selects the single best row from each frontier to form the
    trajectory.

    Attributes
    ----------
    states : States
        Selected indices, shape ``(N, d)``.
    contexts : Contexts
        Per-state energy context, shape ``(N, K)``.
    energies : Energies
        Per-state energy values, shape ``(N,)``. Lower is better.
    """

    states: States
    contexts: Contexts
    energies: Energies

    def __len__(self) -> int:
        return self.states.shape[0]


type Step = Callable[
    [Frontier, np.random.Generator],
    Frontier,
]
"""A single search transition: ``(frontier, rng) -> frontier'``. The transition is responsible for expanding the frontier via a `Neighborhood`, scoring the children with an `Energy`, and selecting which survive."""
