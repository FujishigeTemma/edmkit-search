"""MDE (Manifold Dimension Expansion) greedy variable selection."""

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, TypeAlias

import numpy as np

from .metrics import MetricFn
from .skill import PredictFn, _ensure_2d, prediction_skill
from .splits import TemporalSplit

CandidateFilter: TypeAlias = Callable[
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


class MetricConfig(NamedTuple):
    """Configuration for an MDE metric.

    Parameters
    ----------
    fn : MetricFn
        Metric function.
    threshold : float
        Minimum score for candidate selection.
    """

    fn: MetricFn
    threshold: float


class SelectionStep(NamedTuple):
    """Result of a single dimension selection step.

    Parameters
    ----------
    dim : int
        Dimension number (1-indexed).
    var_idx : int
        Index of the selected variable.
    score : float
        Validation score at this step.
    per_target_scores : np.ndarray
        Per-target validation scores.
    """

    dim: int
    var_idx: int
    score: float
    per_target_scores: np.ndarray


class SelectionResult(NamedTuple):
    """Result of the selection phase (without predictions).

    Parameters
    ----------
    selected_indices : list[int]
        Indices of selected variables (into candidates array).
    val_scores : list[float]
        Mean validation score at each dimension step.
    val_scores_per_target : list[np.ndarray]
        Per-target validation scores at each dimension step.
    """

    selected_indices: list[int]
    val_scores: list[float]
    val_scores_per_target: list[np.ndarray]


class EvaluationResult(NamedTuple):
    """Result of manifold evaluation.

    Parameters
    ----------
    scores : list[float]
        Mean score at each dimension step.
    scores_per_target : list[np.ndarray]
        Per-target scores at each dimension step.
    predictions : list[np.ndarray] | None
        Predictions at each dimension step. None if not requested.
    """

    scores: list[float]
    scores_per_target: list[np.ndarray]
    predictions: list[np.ndarray] | None = None


class MDEResult(NamedTuple):
    """MDE execution result combining selection and evaluation.

    Parameters
    ----------
    split : TemporalSplit
        The train/validation/test split used.
    selected_indices : list[int]
        Indices of selected variables (into candidates array).
    manifold : np.ndarray
        The constructed manifold of shape (T, D) where D is the number of
        selected dimensions.
    validation : EvaluationResult
        Validation set evaluation results.
    test : EvaluationResult
        Test set evaluation results.
    """

    split: TemporalSplit
    selected_indices: list[int]
    manifold: np.ndarray
    validation: EvaluationResult
    test: EvaluationResult


class _EvalCandidateArgs(NamedTuple):
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
    per_target_scores: np.ndarray


def _eval_candidate(args: _EvalCandidateArgs) -> _EvalResult | None:
    """Evaluate a single candidate variable (worker function for parallel execution)."""
    candidate_indices = args.selected_indices + [args.var_idx]
    manifold_train = args.X_train[:, candidate_indices]
    manifold_val = args.X_val[:, candidate_indices]

    predictions = args.predict(manifold_train, args.Y_train, manifold_val)
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    score, per_target_scores = args.metric(predictions, args.Y_val)

    if score < args.threshold:
        return None

    return _EvalResult(
        var_idx=args.var_idx,
        score=score,
        per_target_scores=per_target_scores,
    )


def select_iter(
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
    candidate_filter: CandidateFilter | None = None,
) -> Iterator[SelectionStep]:
    """Greedily select variables that maximize prediction skill (generator version).

    Yields one SelectionStep per dimension, allowing early termination and
    custom control logic.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables where T is number of time points and N is number
        of candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    train_indices : np.ndarray
        Indices for training (library construction).
    val_indices : np.ndarray
        Indices for validation (selection evaluation).
    predict : PredictFn
        Prediction function (X_lib, Y_lib, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : CandidateFilter | None, optional
        Optional filter function to accept/reject candidates. Called on the
        best candidate each iteration; if rejected, tries next best.

    Yields
    ------
    SelectionStep
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
            _EvalCandidateArgs(
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

        yield SelectionStep(
            dim=dim,
            var_idx=best_result.var_idx,
            score=best_result.score,
            per_target_scores=best_result.per_target_scores,
        )


def select(
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
    candidate_filter: CandidateFilter | None = None,
) -> SelectionResult:
    """Greedily select variables that maximize prediction skill.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables where T is number of time points and N is number
        of candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    train_indices : np.ndarray
        Indices for training (library construction).
    val_indices : np.ndarray
        Indices for validation (selection evaluation).
    predict : PredictFn
        Prediction function (X_lib, Y_lib, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.0.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : CandidateFilter | None, optional
        Optional filter function to accept/reject candidates.

    Returns
    -------
    SelectionResult
        Result containing selected_indices, val_scores, and val_scores_per_target.

    Raises
    ------
    ValueError
        If candidates is not 2D, max_dim exceeds N, or shapes mismatch.
    """
    selected_indices: list[int] = []
    val_scores: list[float] = []
    val_scores_per_target: list[np.ndarray] = []

    for step in select_iter(
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
        val_scores.append(step.score)
        val_scores_per_target.append(step.per_target_scores)

    return SelectionResult(
        selected_indices=selected_indices,
        val_scores=val_scores,
        val_scores_per_target=val_scores_per_target,
    )


def get_predictions(
    candidates: np.ndarray,
    target: np.ndarray,
    selected_indices: list[int],
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    predict: PredictFn,
    *,
    dim: int | None = None,
) -> list[np.ndarray]:
    """Compute predictions on-demand for selected manifold dimensions.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    selected_indices : list[int]
        Indices of selected variables.
    train_indices : np.ndarray
        Indices for library construction.
    query_indices : np.ndarray
        Indices for prediction.
    predict : PredictFn
        Prediction function.
    dim : int | None, optional
        If specified, only compute predictions for this dimension (1-indexed).
        Otherwise compute for all dimensions.

    Returns
    -------
    list[np.ndarray]
        Predictions at each dimension step (or single dimension if specified).
    """
    if len(selected_indices) == 0:
        return []

    target = _ensure_2d(target)

    if dim is not None:
        # Single dimension
        if dim < 1 or dim > len(selected_indices):
            raise ValueError(
                f"dim must be between 1 and {len(selected_indices)}, got {dim}"
            )
        manifold = candidates[:, selected_indices[:dim]]
        X_lib = manifold[train_indices]
        Y_lib = target[train_indices]
        X_query = manifold[query_indices]
        preds = predict(X_lib, Y_lib, X_query)
        if preds.ndim == 1:
            preds = preds[:, None]
        return [preds]

    # All dimensions
    predictions: list[np.ndarray] = []
    for i in range(1, len(selected_indices) + 1):
        manifold = candidates[:, selected_indices[:i]]
        X_lib = manifold[train_indices]
        Y_lib = target[train_indices]
        X_query = manifold[query_indices]
        preds = predict(X_lib, Y_lib, X_query)
        if preds.ndim == 1:
            preds = preds[:, None]
        predictions.append(preds)

    return predictions


def evaluate_manifold(
    candidates: np.ndarray,
    target: np.ndarray,
    selected_indices: list[int],
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    predict: PredictFn,
    metric: MetricFn,
    store_predictions: bool = False,
) -> EvaluationResult:
    """Evaluate a manifold on given indices.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    selected_indices : list[int]
        Indices of selected variables.
    train_indices : np.ndarray
        Indices for library construction.
    query_indices : np.ndarray
        Indices for evaluation.
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Metric function.
    store_predictions : bool, optional
        Whether to store predictions. Default is False.

    Returns
    -------
    EvaluationResult
        Evaluation results with optional predictions.
    """
    if len(selected_indices) == 0:
        return EvaluationResult(
            scores=[],
            scores_per_target=[],
            predictions=[] if store_predictions else None,
        )

    target = _ensure_2d(target)
    scores: list[float] = []
    scores_per_target: list[np.ndarray] = []
    predictions: list[np.ndarray] | None = [] if store_predictions else None

    for i in range(1, len(selected_indices) + 1):
        manifold = candidates[:, selected_indices[:i]]
        score, per_target, preds = prediction_skill(
            manifold,
            target,
            train_indices,
            query_indices,
            predict=predict,
            metric=metric,
        )
        scores.append(score)
        scores_per_target.append(per_target)
        if predictions is not None:
            predictions.append(preds)

    return EvaluationResult(
        scores=scores,
        scores_per_target=scores_per_target,
        predictions=predictions,
    )


def mde(
    candidates: np.ndarray,
    target: np.ndarray,
    split: TemporalSplit,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.3,
    max_dim: int = 10,
    max_workers: int | None = None,
    candidate_filter: CandidateFilter | None = None,
    store_predictions: bool = False,
) -> MDEResult:
    """Perform Manifold Dimension Expansion to discover causal relationships.

    MDE greedily selects variables that maximize the given metric on the
    validation set. Supports both single-target and multi-target optimization.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables where T is number of time points and N is number
        of candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s). Shape (T,) for single-target or (T, M) for
        multi-target where M is the number of target dimensions.
    split : TemporalSplit
        Pre-computed train/validation/test split.
    predict : PredictFn
        Prediction function (X_lib, Y_lib, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
        The function must return (aggregate_score, per_target_scores).
        MDE maximizes this score, so use ``negate()`` for metrics
        where lower is better (e.g., RMSE, MAE).
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.3.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : CandidateFilter | None, optional
        Optional filter function to accept/reject candidates. Use
        ``make_ccm_filter()`` for CCM convergence checking.
    store_predictions : bool, optional
        Whether to store predictions for visualization. Default is False.

    Returns
    -------
    MDEResult
        Result containing split, selected_indices, manifold, validation, and test.

    Raises
    ------
    ValueError
        If candidates is not 2D, max_dim exceeds N, or shapes mismatch.

    Examples
    --------
    >>> from mde import mde, mean_rho, temporal_split
    >>> from edmkit import simplex_projection
    >>> split = temporal_split(len(candidates), 0.6, 0.2, gap=50)
    >>> result = mde(
    ...     candidates, target, split,
    ...     predict=simplex_projection, metric=mean_rho, threshold=0.3
    ... )

    Using custom metric (minimize RMSE):

    >>> from mde import negate, rmse
    >>> result = mde(
    ...     candidates, target, split,
    ...     predict=simplex_projection, metric=negate(rmse), threshold=-0.5
    ... )
    """
    T = candidates.shape[0]
    target_2d = _ensure_2d(target)

    # 1. Selection phase (train/val)
    selection = select(
        candidates,
        target_2d,
        split.train,
        split.val,
        predict=predict,
        metric=metric,
        threshold=threshold,
        max_dim=max_dim,
        max_workers=max_workers,
        candidate_filter=candidate_filter,
    )

    if len(selection.selected_indices) == 0:
        return MDEResult(
            split=split,
            selected_indices=[],
            manifold=np.array([]).reshape(T, 0),
            validation=EvaluationResult(
                scores=[],
                scores_per_target=[],
                predictions=[] if store_predictions else None,
            ),
            test=EvaluationResult(
                scores=[],
                scores_per_target=[],
                predictions=[] if store_predictions else None,
            ),
        )

    # 2. Validation evaluation (to get predictions if requested)
    val_result = EvaluationResult(
        scores=selection.val_scores,
        scores_per_target=selection.val_scores_per_target,
        predictions=(
            get_predictions(
                candidates,
                target_2d,
                selection.selected_indices,
                split.train,
                split.val,
                predict,
            )
            if store_predictions
            else None
        ),
    )

    # 3. Test evaluation
    test_result = evaluate_manifold(
        candidates,
        target_2d,
        selection.selected_indices,
        split.train,
        split.test,
        predict=predict,
        metric=metric,
        store_predictions=store_predictions,
    )

    # 4. Build result
    final_manifold = candidates[:, selection.selected_indices]

    return MDEResult(
        split=split,
        selected_indices=selection.selected_indices,
        manifold=final_manifold,
        validation=val_result,
        test=test_result,
    )
