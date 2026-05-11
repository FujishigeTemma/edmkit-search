from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edmkit.search.energy import Contexts, Energies
from edmkit.search.state import States


@dataclass(frozen=True)
class Frontier:
    states: States
    contexts: Contexts
    energies: Energies

    def __len__(self) -> int:
        return self.states.shape[0]


type Step = Callable[
    [Frontier, np.random.Generator],
    Frontier,
]
