from edmkit.search.energy import Energy
from edmkit.search.neighborhood import Neighborhood

from .beam import beam
from .frontier import Strategy


def greedy(
    E: Energy,
    N: Neighborhood,
    *,
    depth: int,
    cutoff: float = float("inf"),
) -> Strategy:
    """Build a greedy `Strategy` that commits to the single best child at each depth.

    Equivalent to `beam` with ``width=1, beams=1`` — a single beam
    that, at every depth, expands only the lowest-energy state
    (subject to ``cutoff``) and never revisits the states it left
    behind.

    Parameters
    ----------
    E : Energy
        Energy used to score children.
    N : Neighborhood
        Neighborhood used to expand parents.
    depth : int
        Number of depths to search below the initial frontier. Must
        be non-negative.
    cutoff : float, default ``float("inf")``
        Children with energy strictly greater than ``cutoff`` are
        discarded before selection. ``inf`` disables the cutoff.

    Returns
    -------
    Strategy
        ``(initial, rng) -> trajectory`` yielding a one-row frontier
        per depth ``1..depth``.
    """
    return beam(E, N, width=1, depth=depth, beams=1, cutoff=cutoff)
