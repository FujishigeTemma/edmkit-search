from collections.abc import Iterator

import numpy as np

from .frontier import Frontier, Step


def run(
    initial: Frontier,
    step: Step,
    *,
    max_steps: int,
    rng: np.random.Generator,
) -> Iterator[Frontier]:
    """Iterate ``step`` from ``initial`` and yield the best survivor of each step.

    On each iteration, ``step`` is applied to the current frontier;
    the row with the lowest energy is yielded as a one-row frontier,
    and the *full* post-step frontier feeds the next iteration. The
    iterator terminates early when ``step`` returns an empty frontier
    (i.e. the search has run out of candidates) or after ``max_steps``
    iterations, whichever comes first.

    Parameters
    ----------
    initial : Frontier
        Starting frontier. For the standard forward-selection setup
        this is the empty state ``state.initial()`` paired with the
        energy's ``initial_ctx``.
    step : Step
        Per-iteration transition (e.g. from `beam` or
        `greedy`).
    max_steps : int
        Upper bound on the number of iterations. Must be non-negative.
    rng : np.random.Generator
        Generator threaded into ``step`` (which in turn passes it to
        the neighborhood) so the whole search is reproducible from a
        single seed.

    Yields
    ------
    Frontier
        One-row frontier — the best survivor of each step. Collecting
        these yields the search trajectory.

    Raises
    ------
    ValueError
        If ``max_steps`` is negative.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.search import energy, neighborhood, state, strategy

    initial_ctx, plan = energy.holdout(...)
    E = parallel(initial_ctx, plan, pool)  # see e2e/synthetic.py
    N = neighborhood.forward(data.X.shape[1])

    initial = strategy.Frontier(
        states=state.initial(),
        contexts=initial_ctx,
        energies=np.array([float("inf")], dtype=np.float64),
    )
    step = strategy.greedy(E, N)

    rng = np.random.default_rng(0)
    trace = list(strategy.run(initial, step, max_steps=8, rng=rng))
    selected = trace[-1].states[0]  # final selected indices
    ```
    """
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")

    frontier = initial
    for _ in range(max_steps):
        frontier = step(frontier, rng)
        if not frontier:
            return
        i = int(np.argmin(frontier.energies))
        yield Frontier(
            states=frontier.states[i : i + 1],
            contexts=frontier.contexts[i : i + 1],
            energies=frontier.energies[i : i + 1],
        )
