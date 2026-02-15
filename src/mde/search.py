"""MDE greedy search algorithm for variable selection."""

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, TypeAlias

import numpy as np

from .metrics import MetricFn
from .skill import PredictFn, _ensure_2d

Filter: TypeAlias = Callable[
    [int, np.ndarray, np.ndarray, list[int]],
    bool,
]
"""Filter function to accept/reject a candidate variable.

Signature: (var_idx, X_train, Y_train, selected_indices) -> accept

Parameters
----------
var_idx : int
    Index of the candidate variable being considered.
X_train : np.ndarray
    Training data for candidates (N_train, N_candidates).
Y_train : np.ndarray
    Training targets (N_train, M).
selected_indices : list[int]
    Currently selected variable indices.

Returns
-------
bool
    True to accept the candidate, False to reject.
"""


class Step(NamedTuple):
    """Result of a single dimension selection step.

    Parameters
    ----------
    dim : int
        Dimension number (1-indexed).
    var_idx : int
        Index of the selected variable.
    score : float
        Validation score at this step.
    """

    dim: int
    var_idx: int
    score: float


class Selection(NamedTuple):
    """Result of the selection phase (without predictions).

    Parameters
    ----------
    selected_indices : list[int]
        Indices of selected variables (into candidates array).
    scores : list[float]
        Score at each dimension step.
    """

    selected_indices: list[int]
    scores: list[float]


class _EvalArgs(NamedTuple):
    """Arguments for _eval_candidate worker function."""

    var_idx: int
    selected_indices: list[int]
    X_train: np.ndarray
    X_val: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    threshold: float
    metric: MetricFn
    predict: PredictFn


class _EvalResult(NamedTuple):
    """Result of evaluating a single candidate variable."""

    var_idx: int
    score: float


def _eval_candidate(args: _EvalArgs) -> _EvalResult | None:
    """Evaluate a single candidate variable (worker function for parallel execution)."""
    candidate_indices = args.selected_indices + [args.var_idx]
    manifold_train = args.X_train[:, candidate_indices]
    manifold_val = args.X_val[:, candidate_indices]

    predictions = args.predict(manifold_train, args.Y_train, manifold_val)
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    score = args.metric(predictions, args.Y_val)

    if score < args.threshold:
        return None

    return _EvalResult(
        var_idx=args.var_idx,
        score=score,
    )


def greedy_iter(
    candidates: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.0,
    max_dim: int = 10,
    max_workers: int | None = None,
    candidate_filter: Filter | None = None,
) -> Iterator[Step]:
    """Greedily select variables that maximize prediction skill (generator version).

    Yields one Step per dimension, allowing early termination and
    custom control logic.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables where T is number of time points and N is number
        of candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    train_indices : np.ndarray
        Indices for training.
    val_indices : np.ndarray
        Indices for validation (selection evaluation).
    predict : PredictFn
        Prediction function (X_train, Y_train, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : Filter | None, optional
        Optional filter function to accept/reject candidates. Called on the
        best candidate each iteration; if rejected, tries next best.

    Yields
    ------
    Step
        Result of each dimension selection step.

    Raises
    ------
    ValueError
        If candidates is not 2D, max_dim exceeds N, or shapes mismatch.
    """
    if candidates.ndim != 2:
        raise ValueError(
            f"candidates must be 2D array, got {candidates.ndim}D "
            f"with shape {candidates.shape}"
        )
    T, N = candidates.shape
    if max_dim > N:
        raise ValueError(f"max_dim must be <= N (={N}), got {max_dim}")

    target = _ensure_2d(target)
    if target.shape[0] != T:
        raise ValueError(
            f"target length ({target.shape[0]}) must match candidates ({T})"
        )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    X_train = candidates[train_indices]
    X_val = candidates[val_indices]
    Y_train = target[train_indices]
    Y_val = target[val_indices]

    available_indices = list(range(N))
    selected_indices: list[int] = []

    for dim in range(1, max_dim + 1):
        eval_args = [
            _EvalArgs(
                var_idx=var_idx,
                selected_indices=selected_indices,
                X_train=X_train,
                X_val=X_val,
                Y_train=Y_train,
                Y_val=Y_val,
                threshold=threshold,
                metric=metric,
                predict=predict,
            )
            for var_idx in available_indices
        ]

        if max_workers > 1 and len(available_indices) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_eval_candidate, eval_args))
        else:
            results = [_eval_candidate(args) for args in eval_args]

        # Sort by score descending (for filter fallback)
        valid_results = [r for r in results if r is not None]
        valid_results.sort(key=lambda r: r.score, reverse=True)

        # Find best candidate that passes filter
        best_result = None
        for result in valid_results:
            if candidate_filter is not None:
                if not candidate_filter(
                    result.var_idx, X_train, Y_train, selected_indices
                ):
                    continue
            best_result = result
            break

        if best_result is None:
            return

        selected_indices.append(best_result.var_idx)
        available_indices.remove(best_result.var_idx)

        yield Step(
            dim=dim,
            var_idx=best_result.var_idx,
            score=best_result.score,
        )


def greedy(
    candidates: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.0,
    max_dim: int = 10,
    max_workers: int | None = None,
    candidate_filter: Filter | None = None,
) -> Selection:
    """Greedily select variables that maximize prediction skill.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables where T is number of time points and N is number
        of candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    train_indices : np.ndarray
        Indices for training.
    val_indices : np.ndarray
        Indices for validation (selection evaluation).
    predict : PredictFn
        Prediction function (X_train, Y_train, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : Filter | None, optional
        Optional filter function to accept/reject candidates.

    Returns
    -------
    Selection
        Result containing selected_indices and scores.

    Raises
    ------
    ValueError
        If candidates is not 2D, max_dim exceeds N, or shapes mismatch.
    """
    selected_indices: list[int] = []
    scores: list[float] = []

    for step in greedy_iter(
        candidates,
        target,
        train_indices,
        val_indices,
        predict=predict,
        metric=metric,
        threshold=threshold,
        max_dim=max_dim,
        max_workers=max_workers,
        candidate_filter=candidate_filter,
    ):
        selected_indices.append(step.var_idx)
        scores.append(step.score)

    return Selection(
        selected_indices=selected_indices,
        scores=scores,
    )
