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
    """Build a multi-fold `Plan` that scores each state as a weighted improvement over the previous step.

    Each state is scored on every fold to produce a per-fold metric
    vector. The energy reported for the state is the ``weight``-ed sum
    of ``(metric - previous_metric)`` across folds, so the search is
    driven by the *delta* relative to the parent state. The per-fold
    metric vector is then carried forward as the new context.

    The weighting function (e.g. `softmax`) is applied to the
    incoming contexts and decides how strongly each fold contributes
    to the energy — a per-fold attention mechanism over the search
    trajectory. A low-temperature softmax focuses energy on the folds
    where the parent state is already strongest, penalizing regression
    there; a high-temperature softmax tends toward an unweighted mean
    across folds.

    Parameters
    ----------
    data : dataset.Dataset
        Dataset whose columns are selected by each state.
    folds : Sequence[Fold]
        Folds to score on. Must be non-empty.
    predict : PredictFunc
        Prediction function with signature ``(X, Y, Q) -> predictions``.
    metric : MetricFunc
        Per-fold reducer. Lower must mean better.
    weight : WeightFunc
        Function ``(N, K) -> (N, K)`` producing per-fold weights from
        the incoming per-fold context.
    batch_size : int, default 10000
        Number of states processed in a single job.

    Returns
    -------
    initial : Contexts
        Initial context of shape ``(1, K)`` filled with zeros, where
        ``K = len(folds)``. The zero baseline means the first step's
        energy is just the weighted metric.
    plan : Plan
        Plan that yields one job per ``batch_size`` chunk of states.

    Raises
    ------
    ValueError
        If ``folds`` is empty.

    Examples
    --------
    ```python
    from edmkit.metrics import mean_rho
    from edmkit.simplex_projection import simplex_projection
    from edmkit.splits import sliding_folds

    from edmkit.search import energy


    def corr(predictions, observations):  # strategies minimize energy
        return 1.0 - mean_rho(predictions.reshape(observations.shape), observations)


    inner_folds = sliding_folds(
        train.X.shape[0],
        train_size=int(train.X.shape[0] * 0.4),
        validation_size=int(train.X.shape[0] * 0.2),
        stride=int(train.X.shape[0] * 0.2),
    )

    initial_context, plan = energy.folds(
        data=train,
        folds=inner_folds,
        predict=simplex_projection,
        metric=corr,
        weight=energy.weight.softmax(temperature=1.0),
    )
    ```
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
