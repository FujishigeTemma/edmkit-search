from collections.abc import Callable, Iterable

import numpy as np
from edmkit.metrics import MetricFunc

from edmkit import simplex_projection
from edmkit.search import dataset
from edmkit.search.state import States

from .energy import Contexts, Plan


def loo(
    *,
    data: dataset.Dataset,
    metric: MetricFunc,
    theiler_window: int = 0,
    batch_size: int = 10000,
) -> tuple[Contexts, Plan]:
    if theiler_window < 0:
        raise ValueError("theiler_window must be non-negative")

    initial = np.empty((1, 0), dtype=np.float64)

    def plan(states: States, _contexts: Contexts) -> Iterable[Callable[[], tuple[slice, np.ndarray, np.ndarray]]]:
        n = states.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            def job(start: int = start, end: int = end) -> tuple[slice, np.ndarray, np.ndarray]:
                size = end - start
                idx = states[start:end]
                X = np.ascontiguousarray(data.X[:, idx].transpose(1, 0, 2))
                Y = np.broadcast_to(data.Y, (size, *data.Y.shape))
                energies = metric(
                    simplex_projection.loo(X, Y, theiler_window=theiler_window),
                    Y,
                )
                return (
                    slice(start, end),
                    energies,
                    np.empty((size, 0), dtype=np.float64),
                )

            yield job

    return initial, plan
