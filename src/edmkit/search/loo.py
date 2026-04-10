from collections.abc import Iterator

import numpy as np
from edmkit.metrics import MetricFunc
from edmkit.simplex_projection import loo

from .dataset import Dataset, Subset
from .types import FilterFn, Step


def theiler_window(E: int, *, tau: int, n_ahead: int) -> int:
    """Compute the Theiler window half-width.

    ``(E - 1) * tau + n_ahead`` ensures that no library point shares
    embedding coordinates or prediction-target time indices with the
    query point.
    """
    return (E - 1) * tau + n_ahead


def loo_score(
    indices: list[int],
    *,
    X: np.ndarray,
    Y: np.ndarray,
    metric: MetricFunc,
    tau: int,
    n_ahead: int,
) -> float:
    """Score a variable subset by LOO prediction quality."""
    predictions = loo(
        X[:, indices],
        Y,
        theiler_window=theiler_window(len(indices), tau=tau, n_ahead=n_ahead),
    )
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    return float(metric(predictions, Y))


def greedy_loo(
    data: Dataset | Subset,
    *,
    metric: MetricFunc,
    tau: int,
    n_ahead: int,
    threshold: float = 0.0,
    max_dim: int = 10,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Greedily select variables using leave-one-out cross-validation.

    Each candidate variable set is scored by computing LOO predictions
    via ``simplex_projection`` with Theiler window exclusion and
    evaluating the metric on all LOO predictions at once.

    The Theiler window ``(E - 1) * tau + n_ahead`` grows with the number
    of selected variables E, preventing temporal data leakage at every
    step.

    Parameters
    ----------
    data : Dataset | Subset
        Training data. ``.X`` is the predictor matrix, ``.Y`` the target.
    metric : MetricFunc
        Evaluation metric (higher is better).
    tau : int
        Time delay used in the embedding.
    n_ahead : int
        Prediction horizon (number of steps ahead).
    threshold : float
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.

    Yields
    ------
    Step
        Result of each dimension selection step.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    X = data.X
    Y = data.Y

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D with shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]
    elif Y.ndim != 2:
        raise ValueError(
            f"Y must be 1D or 2D array, got {Y.ndim}D with shape {Y.shape}"
        )

    M = X.shape[1]
    if max_dim > M:
        raise ValueError(f"max_dim must be <= M (={M}), got {max_dim}")

    available: set[int] = set(range(M))
    selected_indices: list[int] = []

    for _ in range(max_dim):
        results = [
            (
                v,
                loo_score(
                    selected_indices + [v],
                    X=X,
                    Y=Y,
                    metric=metric,
                    tau=tau,
                    n_ahead=n_ahead,
                ),
            )
            for v in available
        ]

        candidates = sorted(
            ((idx, s) for idx, s in results if s >= threshold),
            key=lambda r: r[1],
            reverse=True,
        )

        best = None
        for idx, score in candidates:
            if filter is not None and not filter(X[:, idx], Y):
                continue
            best = (idx, score)
            break

        if best is None:
            return

        best_idx, best_score = best
        selected_indices.append(best_idx)
        available.remove(best_idx)

        yield Step(
            index=best_idx,
            score=best_score,
            selected=tuple(selected_indices),
        )
