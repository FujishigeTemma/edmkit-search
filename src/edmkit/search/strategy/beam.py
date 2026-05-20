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
    """Build a beam-search `Step` that keeps the ``width`` lowest-energy children.

    On each invocation, the step expands the incoming frontier via
    ``N``, scores every child with ``E``, drops children whose energy
    exceeds ``cutoff``, and then keeps the ``width`` survivors with
    the lowest energy. The argsort is stable, so ties resolve in the
    order ``N`` emitted the children — which, for the standard
    `forward` neighborhood, means a per-row random tie-break.

    Parameters
    ----------
    E : Energy
        Energy used to score children. Must already be wired to its
        plan executor (see `Energy`).
    N : Neighborhood
        Neighborhood used to expand parents.
    width : int
        Number of children retained per step. Must be at least 1.
        ``width=1`` reduces this to `greedy`.
    cutoff : float, default ``float("inf")``
        Children with energy strictly greater than ``cutoff`` are
        discarded before truncation. ``inf`` disables the cutoff.

    Returns
    -------
    Step
        ``(frontier, rng) -> frontier'`` returning a frontier of at
        most ``width`` rows. Returns an empty frontier when the
        neighborhood emits no children.

    Raises
    ------
    ValueError
        If ``width < 1``.
    """
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
