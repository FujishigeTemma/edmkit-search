"""MDE high-level pipeline: evaluation and result construction."""

from typing import NamedTuple

import numpy as np

from .metrics import MetricFn
from .search import Filter, Selection, greedy
from .skill import PredictFn, _ensure_2d, prediction_skill
from .splits import Split


class Evaluation(NamedTuple):
    """Result of manifold evaluation.

    Parameters
    ----------
    scores : list[float]
        Score at each dimension step.
    predictions : list[np.ndarray] | None
        Predictions at each dimension step. None if not requested.
    """

    scores: list[float]
    predictions: list[np.ndarray] | None = None


class Result(NamedTuple):
    """MDE execution result combining selection and evaluation.

    Parameters
    ----------
    split : Split
        The train/validation/test split used.
    selected_indices : list[int]
        Indices of selected variables (into candidates array).
    manifold : np.ndarray
        The constructed manifold of shape (T, D) where D is the number of
        selected dimensions.
    val : Evaluation
        Validation set evaluation results.
    test : Evaluation
        Test set evaluation results.
    """

    split: Split
    selected_indices: list[int]
    manifold: np.ndarray
    val: Evaluation
    test: Evaluation


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
        Indices for training.
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
        X_train = manifold[train_indices]
        Y_train = target[train_indices]
        X_query = manifold[query_indices]
        prediction = predict(X_train, Y_train, X_query)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        return [prediction]

    # All dimensions
    predictions: list[np.ndarray] = []
    for i in range(1, len(selected_indices) + 1):
        manifold = candidates[:, selected_indices[:i]]
        X_train = manifold[train_indices]
        Y_train = target[train_indices]
        X_query = manifold[query_indices]
        prediction = predict(X_train, Y_train, X_query)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        predictions.append(prediction)

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
) -> Evaluation:
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
        Indices for training.
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
    Evaluation
        Evaluation results with optional predictions.
    """
    if len(selected_indices) == 0:
        return Evaluation(
            scores=[],
            predictions=[] if store_predictions else None,
        )

    target = _ensure_2d(target)
    scores: list[float] = []
    predictions: list[np.ndarray] | None = [] if store_predictions else None

    for i in range(1, len(selected_indices) + 1):
        manifold = candidates[:, selected_indices[:i]]
        score, prediction = prediction_skill(
            manifold,
            target,
            train_indices,
            query_indices,
            predict=predict,
            metric=metric,
        )
        scores.append(score)
        if predictions is not None:
            predictions.append(prediction)

    return Evaluation(
        scores=scores,
        predictions=predictions,
    )


def build_result(
    candidates: np.ndarray,
    target: np.ndarray,
    split: Split,
    selection: Selection,
    *,
    predict: PredictFn,
    metric: MetricFn,
    store_predictions: bool = False,
) -> Result:
    """Build an MDE Result from a completed selection.

    This function evaluates the selected manifold on validation and test sets,
    allowing any search strategy to be combined with the standard evaluation
    pipeline.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    split : Split
        Pre-computed train/validation/test split.
    selection : Selection
        Result of a search algorithm (e.g., ``greedy()``).
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Metric function.
    store_predictions : bool, optional
        Whether to store predictions. Default is False.

    Returns
    -------
    Result
        Result containing split, selected_indices, manifold, val, and test.
    """
    T = candidates.shape[0]
    target_2d = _ensure_2d(target)

    if len(selection.selected_indices) == 0:
        return Result(
            split=split,
            selected_indices=[],
            manifold=np.array([]).reshape(T, 0),
            val=Evaluation(
                scores=[],
                predictions=[] if store_predictions else None,
            ),
            test=Evaluation(
                scores=[],
                predictions=[] if store_predictions else None,
            ),
        )

    # Validation evaluation
    val_result = Evaluation(
        scores=selection.val_scores,
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

    # Test evaluation
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

    final_manifold = candidates[:, selection.selected_indices]

    return Result(
        split=split,
        selected_indices=selection.selected_indices,
        manifold=final_manifold,
        val=val_result,
        test=test_result,
    )


def mde(
    candidates: np.ndarray,
    target: np.ndarray,
    split: Split,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.3,
    max_dim: int = 10,
    max_workers: int | None = None,
    candidate_filter: Filter | None = None,
    store_predictions: bool = False,
) -> Result:
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
    split : Split
        Pre-computed train/validation/test split.
    predict : PredictFn
        Prediction function (X_train, Y_train, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
        Must return a scalar score. MDE maximizes this score,
        so use ``negate()`` for metrics where lower is better
        (e.g., RMSE, MAE).
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.3.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : Filter | None, optional
        Optional filter function to accept/reject candidates. Use
        ``ccm.make_filter()`` for CCM convergence checking.
    store_predictions : bool, optional
        Whether to store predictions for visualization. Default is False.

    Returns
    -------
    Result
        Result containing split, selected_indices, manifold, val, and test.

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
    selection = greedy(
        candidates,
        _ensure_2d(target),
        split.train,
        split.val,
        predict=predict,
        metric=metric,
        threshold=threshold,
        max_dim=max_dim,
        max_workers=max_workers,
        candidate_filter=candidate_filter,
    )

    return build_result(
        candidates,
        target,
        split,
        selection,
        predict=predict,
        metric=metric,
        store_predictions=store_predictions,
    )
