from collections.abc import Iterator
from functools import partial
from typing import NamedTuple

import numpy as np

from .dataset import Dataset, Subset
from .types import FilterFn, MetricFn, PredictFn


class Evaluation(NamedTuple):
    """Result of evaluating a single candidate variable."""

    index: int
    score: float


def evaluate(
    index: int,
    *,
    selected_indices: list[int],
    X_train: np.ndarray,
    X_validation: np.ndarray,
    Y_train: np.ndarray,
    Y_validation: np.ndarray,
    metric: MetricFn,
    predict: PredictFn,
) -> Evaluation:
    """Evaluate a single candidate variable."""
    candidate_indices = selected_indices + [index]

    predictions = predict(
        X_train[:, candidate_indices], Y_train, X_validation[:, candidate_indices]
    )
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    score = metric(predictions, Y_validation)

    return Evaluation(index=index, score=score)


class Step(NamedTuple):
    """Result of a single dimension selection step.

    Parameters
    ----------
    index : int
        Index of the selected variable.
    score : float
        Validation score at this step.
    """

    index: int
    score: float


def greedy_iter(
    train: Dataset | Subset,
    validation: Dataset | Subset,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.0,
    max_dim: int = 10,
    filter: FilterFn | None = None,
) -> Iterator[Step]:
    """Greedily select variables that maximize prediction skill (generator version).

    Yields one ``Step`` per dimension, allowing early termination and
    custom control logic.

    Parameters
    ----------
    train : Dataset | Subset
        Training data. Only ``.X`` and ``.Y`` are accessed.
    validation : Dataset | Subset
        Validation data for evaluating candidates.
    predict : PredictFn
        Prediction function ``(X_train, Y_train, X_query) -> predictions``.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.
        Called on the best candidate each iteration; if rejected, tries next best.

    Yields
    ------
    Step
        Result of each dimension selection step.
    """
    X_train = train.X
    X_validation = validation.X
    Y_train = train.Y
    Y_validation = validation.Y

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]
    if Y_validation.ndim == 1:
        Y_validation = Y_validation[:, None]

    if X_train.ndim != 2:
        raise ValueError(
            f"X must be 2D array, got {X_train.ndim}D with shape {X_train.shape}"
        )
    N = X_train.shape[1]
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")

    available_indices = list(range(N))
    selected_indices: list[int] = []

    for dim in range(1, max_dim + 1):
        eval = partial(
            evaluate,
            selected_indices=selected_indices,
            X_train=X_train,
            X_validation=X_validation,
            Y_train=Y_train,
            Y_validation=Y_validation,
            metric=metric,
            predict=predict,
        )

        results = [eval(v) for v in available_indices]

        candidates = sorted(
            (r for r in results if r.score >= threshold),
            key=lambda r: r.score,
            reverse=True,
        )

        best = None
        for candidate in candidates:
            if filter is not None and not filter(X_train[:, candidate.index], Y_train):
                continue
            best = candidate
            break

        if best is None:
            return

        selected_indices.append(best.index)
        available_indices.remove(best.index)

        yield Step(index=best.index, score=best.score)


class Selection(NamedTuple):
    """Result of the selection phase.

    Parameters
    ----------
    indices : list[int]
        Indices of selected variables.
    scores : list[float]
        Score at each dimension step.
    """

    indices: list[int]
    scores: list[float]


def greedy(
    train: Dataset | Subset,
    validation: Dataset | Subset,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.0,
    max_dim: int = 10,
    filter: FilterFn | None = None,
) -> Selection:
    """Greedily select variables that maximize prediction skill.

    Parameters
    ----------
    train : Dataset | Subset
        Training data. Only ``.X`` and ``.Y`` are accessed.
    validation : Dataset | Subset
        Validation data for evaluating candidates.
    predict : PredictFn
        Prediction function ``(X_train, Y_train, X_query) -> predictions``.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int
        Maximum number of variables to select. Default is 10.
    filter : FilterFn | None
        Optional filter ``(x, Y) -> bool`` to accept/reject candidates.

    Returns
    -------
    Selection
        Result containing ``indices`` and ``scores``.
    """
    indices: list[int] = []
    scores: list[float] = []

    for step in greedy_iter(
        train,
        validation,
        predict=predict,
        metric=metric,
        threshold=threshold,
        max_dim=max_dim,
        filter=filter,
    ):
        indices.append(step.index)
        scores.append(step.score)

    return Selection(
        indices=indices,
        scores=scores,
    )
