from collections.abc import Iterator

import numpy as np

from edmkit.search.energy import Energy
from edmkit.search.neighborhood import Neighborhood

from .frontier import Frontier, Strategy


def beam(
    E: Energy,
    N: Neighborhood,
    *,
    width: int,
    depth: int,
    beams: int,
    cutoff: float = float("inf"),
) -> Strategy:
    """Build a chokudai-search `Strategy`.

    Chokudai search keeps one candidate queue per depth, seeded with
    the initial frontier at depth 0. Each *beam* is one pass over the
    depths in order: it pops the ``width`` lowest-energy states from
    the depth-``d`` queue, expands them via ``N``, scores the children
    with ``E``, and pushes the survivors (those with energy at most
    ``cutoff``) into the depth-``d+1`` queue. Popped states never
    return, so each additional beam expands the next-best states left
    behind by earlier beams — a single beam is exactly classic beam
    search of width ``width``, and every extra beam widens the search
    around the depths where the earlier ones committed.

    Within one depth, ties in the pop order resolve stably in
    insertion order — which, for the standard `forward` neighborhood,
    means a per-row random tie-break.

    Parameters
    ----------
    E : Energy
        Energy used to score children. Must already be wired to its
        plan executor (see `Energy`).
    N : Neighborhood
        Neighborhood used to expand parents.
    width : int
        Number of states popped per depth per beam. Must be at
        least 1.
    depth : int
        Number of depths to search below the initial frontier. Must
        be non-negative.
    beams : int
        Number of passes over the depth queues. Must be at least 1.
        ``beams=1`` reduces this to classic beam search of width
        ``width``; ``width=1, beams=1`` reduces it to `greedy`.
    cutoff : float, default ``float("inf")``
        Children with energy strictly greater than ``cutoff`` are
        discarded and never enqueued. ``inf`` disables the cutoff.

    Returns
    -------
    Strategy
        ``(initial, rng) -> trajectory`` yielding a one-row frontier
        per depth ``1..depth`` — the lowest-energy state found at that
        depth across all beams. Iteration stops early at the first
        depth the search never reached (e.g. when the neighborhood
        emits no children). The ``rng`` is threaded into ``N`` so the
        whole search is reproducible from a single seed.

    Raises
    ------
    ValueError
        If ``width < 1``, ``depth < 0``, or ``beams < 1``.
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if beams < 1:
        raise ValueError(f"beams must be >= 1, got {beams}")

    def search(initial: Frontier, rng: np.random.Generator) -> Iterator[Frontier]:
        queues: list[Frontier | None] = [initial, *([None] * depth)]
        best: list[Frontier | None] = [None] * (depth + 1)

        for _ in range(beams):
            for d in range(depth):
                queue = queues[d]
                if queue is None or len(queue) == 0:
                    continue

                order = np.argsort(queue.energies, kind="stable")
                top, rest = order[:width], order[width:]
                queues[d] = Frontier(
                    states=queue.states[rest],
                    contexts=queue.contexts[rest],
                    energies=queue.energies[rest],
                )

                children, parents = N(queue.states[top], rng)
                if children.shape[0] == 0:
                    continue

                energies, contexts = E(children, queue.contexts[top][parents])
                kept = np.flatnonzero(energies <= cutoff)
                if kept.size == 0:
                    continue

                child = Frontier(
                    states=children[kept],
                    contexts=contexts[kept],
                    energies=energies[kept],
                )
                following = queues[d + 1]
                queues[d + 1] = (
                    child
                    if following is None
                    else Frontier(
                        states=np.concatenate([following.states, child.states], axis=0),
                        contexts=np.concatenate([following.contexts, child.contexts], axis=0),
                        energies=np.concatenate([following.energies, child.energies]),
                    )
                )

                i = int(np.argmin(child.energies))
                incumbent = best[d + 1]
                if incumbent is None or child.energies[i] < incumbent.energies[0]:
                    best[d + 1] = Frontier(
                        states=child.states[i : i + 1],
                        contexts=child.contexts[i : i + 1],
                        energies=child.energies[i : i + 1],
                    )

        for d in range(1, depth + 1):
            found = best[d]
            if found is None:
                return
            yield found

    return search
