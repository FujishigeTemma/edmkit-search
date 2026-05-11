from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from edmkit.search import dataset
from edmkit.search.state import States

from .energy import Contexts, Energies, Energy
from .weight import WeightFunc

BATCH_SIZE = 10000


def folds(
    *,
    data: dataset.Dataset,
    folds: Sequence[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    weight: WeightFunc,
) -> Energy:
    if len(folds) == 0:
        raise ValueError("folds must be non-empty")

    n_folds = len(folds)
    subsets = [
        (dataset.Subset(data, f.train), dataset.Subset(data, f.validation))
        for f in folds
    ]

    def initial() -> npt.NDArray[np.float64]:
        return np.zeros((1, n_folds), dtype=np.float64)

    def step(
        states: States,
        contexts: Contexts,
    ) -> tuple[Energies, Contexts]:
        n = states.shape[0]
        metrics = np.empty((n, n_folds), dtype=np.float64)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            size = end - start
            idx = states[start:end].astype(np.intp)
            for i, (train, validation) in enumerate(subsets):
                X = np.ascontiguousarray(train.X[:, idx].transpose(1, 0, 2))
                Y = np.broadcast_to(train.Y, (size, *train.Y.shape))
                Q = np.ascontiguousarray(validation.X[:, idx].transpose(1, 0, 2))
                metrics[start:end, i] = metric(
                    predict(X, Y, Q),
                    np.broadcast_to(validation.Y, (size, *validation.Y.shape)),
                )

        energies = np.array(
            [weight(contexts[i]) @ (metrics[i] - contexts[i]) for i in range(n)],
            dtype=np.float64,
        )

        return energies, metrics

    return Energy(initial=initial, step=step)
