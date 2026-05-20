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
    """Build a greedy `Step` that keeps only the single best child per parent.

    Equivalent to `beam` with ``width=1`` — the lowest-energy
    child (subject to ``cutoff``) replaces the frontier on each step.

    Parameters
    ----------
    E : Energy
        Energy used to score children.
    N : Neighborhood
        Neighborhood used to expand parents.
    cutoff : float, default ``float("inf")``
        Children with energy strictly greater than ``cutoff`` are
        discarded before selection.

    Returns
    -------
    Step
        ``(frontier, rng) -> frontier'`` returning a frontier of at
        most one row.
    """
    return beam(E, N, width=1, cutoff=cutoff)
