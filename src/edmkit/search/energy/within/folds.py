from collections.abc import Callable, Iterable, Sequence

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from edmkit.search import dataset
from edmkit.search.state import States

from ..energy import Contexts, Plan
from ..weight import WeightFunc


def folds(
    *,
    data: dataset.Dataset,
    folds: Sequence[Fold],
    predict: PredictFunc,
    metric: MetricFunc,
    weight: WeightFunc,
    batch_size: int = 10000,
) -> tuple[Contexts, Plan]:
    """Build a multi-fold `Plan` for within-state prediction.

    This is the within-state analogue of ``energy.cross.folds``: each
    state is scored on every fold, and the reported energy is the
    weighted delta from the incoming per-fold context. Each state
    forecasts its own selected columns: ``X[:, state] -> Y[:, state]``.

    Parameters
    ----------
    data : dataset.Dataset
        Dataset whose ``X`` columns are candidate state coordinates.
    folds : Sequence[Fold]
        Folds to score on. Must be non-empty.
    predict : PredictFunc
        Prediction function with signature ``(X, Y, Q) -> predictions``.
    metric : MetricFunc
        Per-fold reducer. Lower must mean better.
    weight : WeightFunc
        Function ``(N, K) -> (N, K)`` producing per-fold weights from
        incoming contexts.
    batch_size : int, default 10000
        Number of states processed in a single job.

    Returns
    -------
    initial : Contexts
        Initial context of shape ``(1, K)`` filled with zeros, where
        ``K = len(folds)``.
    plan : Plan
        Plan that yields one job per ``batch_size`` chunk of states.

    Raises
    ------
    ValueError
        If ``folds`` is empty.
    """
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
                    Y = np.ascontiguousarray(train.Y[:, idx].transpose(1, 0, 2))
                    Q = np.ascontiguousarray(validation.X[:, idx].transpose(1, 0, 2))
                    metrics[:, i] = metric(predict(X, Y, Q), np.ascontiguousarray(validation.Y[:, idx].transpose(1, 0, 2)))
                return (
                    slice(start, end),
                    (weight(contexts[start:end]) * (metrics - contexts[start:end])).sum(axis=1),
                    metrics,
                )

            yield job

    return initial, plan
