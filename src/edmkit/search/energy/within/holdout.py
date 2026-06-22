from collections.abc import Callable, Iterable

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from edmkit.search import dataset
from edmkit.search.state import States

from ..energy import Contexts, Plan


def holdout(
    *,
    data: dataset.Dataset,
    fold: Fold,
    predict: PredictFunc,
    metric: MetricFunc,
    batch_size: int = 10000,
) -> tuple[Contexts, Plan]:
    """Build a holdout-validation `Plan` for within-state prediction.

    For each state, the selected columns of ``data.X`` are the library/
    query coordinates and the *same* selected columns of ``data.Y`` are
    the prediction target: ``X[:, state] -> Y[:, state]``.

    Parameters
    ----------
    data : dataset.Dataset
        Dataset whose ``X`` columns are candidate state coordinates.
    fold : Fold
        Train/validation split used to score every state.
    predict : PredictFunc
        Prediction function with signature ``(X, Y, Q) -> predictions``.
    metric : MetricFunc
        Reducer turning predictions and observations into a scalar per
        state. Lower must mean better.
    batch_size : int, default 10000
        Number of states processed in a single job.

    Returns
    -------
    initial : Contexts
        Initial context of shape ``(1, 0)``.
    plan : Plan
        Plan that yields one job per ``batch_size`` chunk of states.
    """
    train = dataset.Subset(data, fold.train)
    validation = dataset.Subset(data, fold.validation)
    initial = np.empty((1, 0), dtype=np.float64)

    def plan(states: States, _contexts: Contexts) -> Iterable[Callable[[], tuple[slice, np.ndarray, np.ndarray]]]:
        n = states.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            def job(start: int = start, end: int = end) -> tuple[slice, np.ndarray, np.ndarray]:
                size = end - start
                idx = states[start:end]
                X = np.ascontiguousarray(train.X[:, idx].transpose(1, 0, 2))
                Y = np.ascontiguousarray(train.Y[:, idx].transpose(1, 0, 2))
                Q = np.ascontiguousarray(validation.X[:, idx].transpose(1, 0, 2))
                energies = metric(predict(X, Y, Q), np.ascontiguousarray(validation.Y[:, idx].transpose(1, 0, 2)))
                return (
                    slice(start, end),
                    energies,
                    np.empty((size, 0), dtype=np.float64),
                )

            yield job

    return initial, plan
