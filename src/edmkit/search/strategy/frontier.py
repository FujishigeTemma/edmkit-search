from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from edmkit.search.energy import Contexts, Energies
from edmkit.search.state import States


@dataclass(frozen=True)
class Frontier:
    """Immutable batch of search candidates.

    The three arrays are aligned along their leading axis: row ``i`` of
    ``states`` corresponds to ``contexts[i]`` and ``energies[i]``. A
    `Strategy` consumes an initial frontier and yields one-row
    frontiers — the best state found at each depth — as the search
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


type Strategy = Callable[
    [Frontier, np.random.Generator],
    Iterator[Frontier],
]
"""A full search: ``(initial, rng) -> trajectory``. Given the starting frontier, the strategy owns the whole loop — expanding states via a `Neighborhood`, scoring them with an `Energy`, and deciding which survive — and yields the best state found at each depth as a one-row frontier. Collecting the iterator yields the search trajectory."""
