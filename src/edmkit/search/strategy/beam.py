import numpy as np

from edmkit.search.energy import Energy
from edmkit.search.neighborhood import Neighborhood

from .frontier import Frontier, Step


def beam(
    E: Energy,
    N: Neighborhood,
    *,
    width: int,
    cutoff: float = float("inf"),
) -> Step:
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    def step(frontier: Frontier, rng: np.random.Generator) -> Frontier:
        children, parents = N(frontier.states, rng)
        if children.shape[0] == 0:
            return Frontier(
                states=children,
                contexts=frontier.contexts[:0],
                energies=np.empty(0, dtype=np.float64),
            )

        energies, contexts = E(children, frontier.contexts[parents])
        kept = np.flatnonzero(energies <= cutoff)
        order = kept[np.argsort(energies[kept], kind="stable")[:width]]
        return Frontier(
            states=children[order],
            contexts=contexts[order],
            energies=energies[order],
        )

    return step
