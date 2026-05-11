from edmkit.search.energy import Energy
from edmkit.search.neighborhood import Neighborhood
from edmkit.search.strategy.beam import beam
from edmkit.search.strategy.frontier import Step


def greedy(
    E: Energy,
    N: Neighborhood,
    *,
    cutoff: float = float("inf"),
) -> Step:
    return beam(E, N, width=1, cutoff=cutoff)
