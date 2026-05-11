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
