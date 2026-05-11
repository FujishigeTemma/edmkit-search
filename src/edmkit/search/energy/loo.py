import numpy as np
import numpy.typing as npt
from edmkit.metrics import MetricFunc

from edmkit import simplex_projection
from edmkit.search import dataset
from edmkit.search.state import States

from .energy import Contexts, Energies, Energy

BATCH_SIZE = 10000


def loo(
    *,
    data: dataset.Dataset,
    metric: MetricFunc,
    theiler_window: int = 0,
    max_batch_size: int | None = None,
) -> Energy:
    if theiler_window < 0:
        raise ValueError("theiler_window must be non-negative")

    def initial() -> npt.NDArray[np.float64]:
        return np.empty((1, 0), dtype=np.float64)

    def step(
        states: States,
        contexts: Contexts,
    ) -> tuple[Energies, Contexts]:
        del contexts
        n = states.shape[0]
        energies = np.empty(n, dtype=np.float64)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            size = end - start
            idx = states[start:end].astype(np.intp)
            X = np.ascontiguousarray(data.X[:, idx].transpose(1, 0, 2))
            Y = np.broadcast_to(data.Y, (size, *data.Y.shape))
            energies[start:end] = metric(
                simplex_projection.loo(X, Y, theiler_window=theiler_window),
                Y,
            )
        return energies, np.empty((n, 0), dtype=np.float64)

    return Energy(initial=initial, step=step)
