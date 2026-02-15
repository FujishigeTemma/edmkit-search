"""MDE high-level pipeline: evaluation and result construction."""

from typing import NamedTuple

import numpy as np

from .dataset import Dataset, Fold, Subset, make_expanding_windows
from .metrics import MetricFn
from .search import Filter, Selection, greedy
from .skill import PredictFn, _ensure_2d, prediction_skill


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
    selected_indices : list[int]
        Indices of selected variables (into candidates array).
    manifold : np.ndarray
        The constructed manifold of shape (T, D).
    train_indices : np.ndarray
        All training indices.
    val_indices : np.ndarray
        Evaluated validation indices (fold.val concatenated).
    train_scores : list[float]
        Per-dimension scores from greedy search.
    val : Evaluation
        Expanding window validation results.
    """

    selected_indices: list[int]
    manifold: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_scores: list[float]
    val: Evaluation


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


def evaluate_expanding(
    candidates: np.ndarray,
    target: np.ndarray,
    selected_indices: list[int],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    predict: PredictFn,
    metric: MetricFn,
    folds: list[Fold],
    store_predictions: bool = False,
) -> tuple[Evaluation, np.ndarray]:
    """Expanding window validation.

    For each fold:
      library = train_indices + val_indices[fold.train]
      query = val_indices[fold.val]
      simplex_projection(library -> query)

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        Candidate variables.
    target : np.ndarray of shape (T,) or (T, M)
        Target variable(s).
    selected_indices : list[int]
        Indices of selected variables.
    train_indices : np.ndarray
        Full training indices.
    val_indices : np.ndarray
        Validation indices to split with folds.
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Metric function.
    folds : list[Fold]
        Expanding window folds (indices into val_indices).
    store_predictions : bool, optional
        Whether to store predictions. Default is False.

    Returns
    -------
    evaluation : Evaluation
        scores: per-dimension scores (fold average)
        predictions: per-dimension predictions (fold concatenated)
    query_indices : np.ndarray
        Actually evaluated indices (fold.val concatenated).
    """
    if len(selected_indices) == 0:
        return (
            Evaluation(
                scores=[],
                predictions=[] if store_predictions else None,
            ),
            np.array([], dtype=int),
        )

    target_2d = _ensure_2d(target)

    # Collect all query indices across folds
    all_query_idx = np.concatenate([val_indices[fold.val] for fold in folds])

    n_dims = len(selected_indices)
    dim_scores: list[float] = []
    dim_predictions: list[np.ndarray] | None = [] if store_predictions else None

    for d in range(1, n_dims + 1):
        fold_scores: list[float] = []
        fold_preds: list[np.ndarray] = []

        for fold in folds:
            lib = np.concatenate([train_indices, val_indices[fold.train]])
            query = val_indices[fold.val]

            manifold = candidates[:, selected_indices[:d]]
            score, preds = prediction_skill(
                manifold,
                target_2d,
                lib,
                query,
                predict=predict,
                metric=metric,
            )
            fold_scores.append(score)
            fold_preds.append(preds)

        dim_scores.append(float(np.mean(fold_scores)))
        if dim_predictions is not None:
            dim_predictions.append(np.concatenate(fold_preds, axis=0))

    return (
        Evaluation(scores=dim_scores, predictions=dim_predictions),
        all_query_idx,
    )


def build_result(
    dataset: Dataset,
    train: Subset,
    val: Subset,
    selection: Selection,
    *,
    predict: PredictFn,
    metric: MetricFn,
    store_predictions: bool = False,
    val_folds: list[Fold] | None = None,
) -> Result:
    """Build an MDE Result from a completed selection.

    Parameters
    ----------
    dataset : Dataset
        The full dataset.
    train : Subset
        Training subset.
    val : Subset
        Validation subset.
    selection : Selection
        Result of a search algorithm (e.g., ``greedy()``).
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Metric function.
    store_predictions : bool, optional
        Whether to store predictions. Default is False.
    val_folds : list[Fold] | None, optional
        Expanding window folds for validation. If None, auto-generated.

    Returns
    -------
    Result
    """
    T = len(dataset)

    if len(selection.selected_indices) == 0:
        return Result(
            selected_indices=[],
            manifold=np.array([]).reshape(T, 0),
            train_indices=train.indices,
            val_indices=np.array([], dtype=int),
            train_scores=[],
            val=Evaluation(
                scores=[],
                predictions=[] if store_predictions else None,
            ),
        )

    # Default val folds: expanding window on val data
    if val_folds is None:
        val_folds = make_expanding_windows(
            len(val),
            min_train=0,
            val_size=max(1, len(val) // 5),
        )

    # Expanding window validation
    val_eval, query_idx = evaluate_expanding(
        dataset.X,
        dataset.Y,
        selection.selected_indices,
        train.indices,
        val.indices,
        predict=predict,
        metric=metric,
        folds=val_folds,
        store_predictions=store_predictions,
    )

    final_manifold = dataset.X[:, selection.selected_indices]

    return Result(
        selected_indices=selection.selected_indices,
        manifold=final_manifold,
        train_indices=train.indices,
        val_indices=query_idx,
        train_scores=selection.scores,
        val=val_eval,
    )


def mde(
    dataset: Dataset,
    train: Subset,
    val: Subset,
    *,
    predict: PredictFn,
    metric: MetricFn,
    threshold: float = 0.3,
    max_dim: int = 10,
    max_workers: int | None = None,
    candidate_filter: Filter | None = None,
    store_predictions: bool = False,
    train_folds: list[Fold] | None = None,
    val_folds: list[Fold] | None = None,
) -> Result:
    """Perform Manifold Dimension Expansion to discover causal relationships.

    MDE greedily selects variables that maximize the given metric on the
    training set. Then evaluates on validation with expanding windows.

    Parameters
    ----------
    dataset : Dataset
        The full dataset.
    train : Subset
        Training subset.
    val : Subset
        Validation subset.
    predict : PredictFn
        Prediction function (X_train, Y_train, X_query) -> predictions.
    metric : MetricFn
        Metric function for evaluating prediction quality.
    threshold : float, optional
        Minimum score for candidate selection. Default is 0.3.
    max_dim : int, optional
        Maximum manifold dimension. Default is 10.
    max_workers : int | None, optional
        Maximum number of worker threads. None uses os.cpu_count().
    candidate_filter : Filter | None, optional
        Optional filter function to accept/reject candidates.
    store_predictions : bool, optional
        Whether to store predictions for visualization. Default is False.
    train_folds : list[Fold] | None, optional
        Folds within training data for greedy search. If None, uses a
        single 75/25 split.
    val_folds : list[Fold] | None, optional
        Expanding window folds for validation. If None, auto-generated.

    Returns
    -------
    Result
    """
    # Train phase: greedy search
    if train_folds is None:
        # Default: single 75/25 split within train data
        n_train = len(train)
        split_point = int(n_train * 0.75)
        train_folds = [
            Fold(
                train=np.arange(split_point),
                val=np.arange(split_point, n_train),
            )
        ]

    # Use the last fold for greedy search
    fold = train_folds[-1]
    search_lib = train.indices[fold.train]
    search_eval = train.indices[fold.val]

    selection = greedy(
        dataset.X,
        _ensure_2d(dataset.Y),
        search_lib,
        search_eval,
        predict=predict,
        metric=metric,
        threshold=threshold,
        max_dim=max_dim,
        max_workers=max_workers,
        candidate_filter=candidate_filter,
    )

    return build_result(
        dataset,
        train,
        val,
        selection,
        predict=predict,
        metric=metric,
        store_predictions=store_predictions,
        val_folds=val_folds,
    )
