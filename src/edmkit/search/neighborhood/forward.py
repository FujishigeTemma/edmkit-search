import numpy as np
import numpy.typing as npt

from edmkit.search.state import States

from .neighborhood import Neighborhood


def forward(n: int) -> Neighborhood:
    """Build a forward-selection `Neighborhood` over ``n`` candidate indices.

    Each parent state of length ``d`` (assumed to hold unique indices
    in ``[0, n)``) expands into exactly ``n - d`` children — one per
    index not yet selected. Within a single parent, the order of the
    emitted children is randomized via the supplied generator so that
    downstream truncations (e.g. beam ``width``) do not systematically
    favour low indices; across parents, the parent order is preserved.

    Parameters
    ----------
    n : int
        Size of the candidate universe. Must be non-negative.

    Returns
    -------
    Neighborhood
        ``(parents, rng) -> (children, parents_idx)`` with
        ``children.shape == (N * (n - d), d + 1)``.

    Raises
    ------
    ValueError
        If ``n`` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    def expand(states: States, rng: np.random.Generator) -> tuple[States, npt.NDArray[np.int64]]:
        N, d = states.shape
        per_parent = n - d
        if N == 0 or per_parent == 0:
            return (
                np.empty((0, d + 1), dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        # Per-row random permutation of [0, n).
        perms = rng.permuted(np.tile(np.arange(n, dtype=np.int64), (N, 1)), axis=1)
        # mask[i, j] = True iff j is not already in states[i].
        mask = np.ones((N, n), dtype=bool)
        mask[np.arange(N)[:, None], states] = False
        # Each row of `perms` has exactly `per_parent` survivors (states are unique).
        survivors = np.take_along_axis(mask, perms, axis=1)
        chosen = perms[survivors].reshape(N, per_parent)

        parents = np.repeat(np.arange(N, dtype=np.int64), per_parent)
        prefix = np.repeat(states, per_parent, axis=0)
        children = np.concatenate([prefix, chosen.reshape(-1, 1)], axis=1)
        return children, parents

    return expand
