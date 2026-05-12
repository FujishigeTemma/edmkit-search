from collections.abc import Callable, Iterable, Sequence

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from edmkit.search import dataset
from edmkit.search.state import States

from .energy import Contexts, Plan
from .weight import WeightFunc


def folds(
    *,
    data: dataset.Dataset,
    folds: Sequence[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    weight: WeightFunc,
    batch_size: int = 10000,
) -> tuple[Contexts, Plan]:
    if len(folds) == 0:
        raise ValueError("folds must be non-empty")

    n_folds = len(folds)
    subsets = [(dataset.Subset(data, f.train), dataset.Subset(data, f.validation)) for f in folds]
    initial = np.zeros((1, n_folds), dtype=np.float64)

    def plan(states: States, contexts: Contexts) -> Iterable[Callable[[], tuple[slice, np.ndarray, np.ndarray]]]:
        n = states.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            def job(start: int = start, end: int = end) -> tuple[slice, np.ndarray, np.ndarray]:
                size = end - start
                idx = states[start:end]
                metrics = np.empty((size, n_folds), dtype=np.float64)
                for i, (train, validation) in enumerate(subsets):
                    X = np.ascontiguousarray(train.X[:, idx].transpose(1, 0, 2))
                    Y = np.broadcast_to(train.Y, (size, *train.Y.shape))
                    Q = np.ascontiguousarray(validation.X[:, idx].transpose(1, 0, 2))
                    metrics[:, i] = metric(
                        predict(X, Y, Q),
                        np.broadcast_to(validation.Y, (size, *validation.Y.shape)),
                    )
                return (
                    slice(start, end),
                    (weight(contexts[start:end]) * (metrics - contexts[start:end])).sum(axis=1),
                    metrics,
                )

            yield job

    return initial, plan
