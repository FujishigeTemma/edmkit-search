from collections.abc import Callable, Iterable

import numpy as np
from edmkit.metrics import MetricFunc

from edmkit import simplex_projection
from edmkit.search import dataset
from edmkit.search.state import States

from ..energy import Contexts, Plan


def loo(
    *,
    data: dataset.Dataset,
    metric: MetricFunc,
    theiler_window: int = 0,
    batch_size: int = 10000,
) -> tuple[Contexts, Plan]:
    """Build a cross-target leave-one-out `Plan` that scores each state.

    For each state in the batch, the corresponding column-subset of
    ``data.X`` is used as the library for simplex-projection LOO; each
    library point is predicted from its in-library neighbours
    (excluding temporally close points via the Theiler window), and
    the predictions are scored against ``data.Y`` with ``metric``.

    Parameters
    ----------
    data : dataset.Dataset
        Dataset whose columns are selected by each state.
    metric : MetricFunc
        Reducer turning predictions and observations into a scalar per
        state. Lower must mean better.
    theiler_window : int, default 0
        Theiler window half-width passed to ``simplex_projection.loo``.
    batch_size : int, default 10000
        Number of states processed in a single job.

    Returns
    -------
    initial : Contexts
        Initial context of shape ``(1, 0)``.
    plan : Plan
        Plan that yields one job per ``batch_size`` chunk of states.

    Raises
    ------
    ValueError
        If ``theiler_window`` is negative.
    """
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
