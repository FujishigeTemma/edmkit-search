from collections.abc import Callable, Iterable

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.splits import Fold
from edmkit.types import PredictFunc

from edmkit.search import dataset
from edmkit.search.state import States

from .energy import Contexts, Plan


def holdout(
    *,
    data: dataset.Dataset,
    fold: Fold,
    predict: PredictFunc,
    metric: MetricFunc,
    batch_size: int = 10000,
) -> tuple[Contexts, Plan]:
    """Build a holdout-validation `Plan` that scores each state on a single fold.

    For each state in the batch, the columns of ``data.X`` indexed by
    the state are used to fit ``predict`` on the fold's train arm and
    score it against the fold's validation arm via ``metric``. The
    metric value becomes the state's energy directly; no carry-over
    context is needed.

    Parameters
    ----------
    data : dataset.Dataset
        Dataset whose columns (``X[:, state]``) are selected by each state.
    fold : Fold
        Train/validation split used to score every state.
    predict : PredictFunc
        Prediction function with signature ``(X, Y, Q) -> predictions``
        (e.g. ``simplex_projection`` or ``partial(smap, theta=...)``).
    metric : MetricFunc
        Reducer turning ``(predictions, observations)`` into a scalar
        per state. Lower must mean better — see the project README on
        framing scores as energies.
    batch_size : int, default 10000
        Number of states processed in a single job. Smaller values
        reduce peak memory; larger values reduce dispatch overhead.

    Returns
    -------
    initial : Contexts
        Initial context of shape ``(1, 0)`` — holdout carries no
        per-state state across steps.
    plan : Plan
        Plan that yields one job per ``batch_size`` chunk of states.

    Examples
    --------
    ```python
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from edmkit.metrics import mean_rho
    from edmkit.simplex_projection import simplex_projection
    from edmkit.splits import temporal_fold

    from edmkit.search import energy, state


    def corr(predictions, observations):  # strategies minimize energy
        return 1.0 - mean_rho(predictions.reshape(observations.shape), observations)


    fold2 = temporal_fold(train.X.shape[0], 0.75)
    initial_context, plan = energy.holdout(
        data=train,
        fold=fold2,
        predict=simplex_projection,
        metric=corr,
        batch_size=64,
    )

    with ThreadPoolExecutor() as pool:
        def E(states: state.States, contexts: energy.Contexts) -> tuple[energy.Energies, energy.Contexts]:
            futures = [pool.submit(job) for job in plan(states, contexts)]
            n = states.shape[0]
            energies = np.empty(n, dtype=np.float64)
            new_contexts = np.empty((n, initial_context.shape[1]), dtype=np.float64)
            for f in futures:
                s, e, c = f.result()
                energies[s] = e
                new_contexts[s] = c
            return energies, new_contexts
        # ... strategy.run(...) inside the with-block
    ```
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
                Y = np.broadcast_to(train.Y, (size, *train.Y.shape))
                Q = np.ascontiguousarray(validation.X[:, idx].transpose(1, 0, 2))
                energies = metric(
                    predict(X, Y, Q),
                    np.broadcast_to(validation.Y, (size, *validation.Y.shape)),
                )
                return (
                    slice(start, end),
                    energies,
                    np.empty((size, 0), dtype=np.float64),
                )

            yield job

    return initial, plan
