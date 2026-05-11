import numpy as np
import numpy.typing as npt
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from edmkit.search import dataset
from edmkit.search.state import States

from .energy import Contexts, Energies, Energy

BATCH_SIZE = 10000


def holdout(
    *,
    data: dataset.Dataset,
    fold: Fold,
    predict: PredictFunc,
    metric: MetricFunc,
) -> Energy:
    train = dataset.Subset(data, fold.train)
    validation = dataset.Subset(data, fold.validation)

    def initial() -> npt.NDArray[np.float64]:
        return np.empty((1, 0), dtype=np.float64)

    def step(
        states: States,
        contexts: Contexts,  # not used
    ) -> tuple[Energies, Contexts]:
        n = states.shape[0]
        energies = np.empty(n, dtype=np.float64)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            size = end - start
            idx = states[start:end].astype(np.intp)
            X = np.ascontiguousarray(train.X[:, idx].transpose(1, 0, 2))
            Y = np.broadcast_to(train.Y, (size, *train.Y.shape))
            Q = np.ascontiguousarray(validation.X[:, idx].transpose(1, 0, 2))
            energies[start:end] = metric(
                predict(X, Y, Q),
                np.broadcast_to(validation.Y, (size, *validation.Y.shape)),
            )
        return energies, np.empty((n, 0), dtype=np.float64)

    return Energy(initial=initial, step=step)
