"""Prediction skill computation for manifold evaluation."""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from .metrics import MetricFn

PredictFn: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""Prediction function signature: (X_lib, Y_lib, X_query) -> predictions."""


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D.

    Converts a 1D array of shape (T,) to 2D array of shape (T, 1).

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (T,) or (T, M).

    Returns
    -------
    np.ndarray
        Array of shape (T, M) where M >= 1.
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D array, got {arr.ndim}D with shape {arr.shape}"
        )
    return arr


def prediction_skill(
    manifold: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    predict: PredictFn,
    metric: MetricFn,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute prediction skill of a manifold.

    Parameters
    ----------
    manifold : np.ndarray of shape (T, D)
        State space manifold.
    targets : np.ndarray of shape (T,) or (T, M)
        Target variable(s) to predict.
    train_indices : np.ndarray
        Indices for library construction.
    query_indices : np.ndarray
        Indices for prediction evaluation.
    predict : PredictFn
        Prediction function.
    metric : MetricFn
        Skill metric function.

    Returns
    -------
    score : float
        Aggregate score across all targets.
    per_target : np.ndarray of shape (M,)
        Score for each target.
    predictions : np.ndarray
        Prediction results.

    Raises
    ------
    ValueError
        If manifold is not 2D or indices are out of bounds.
    """
    if manifold.ndim != 2:
        raise ValueError(
            f"manifold must be 2D, got {manifold.ndim}D with shape {manifold.shape}"
        )
    T = manifold.shape[0]

    if train_indices.ndim != 1 or query_indices.ndim != 1:
        raise ValueError("indices must be 1D arrays")

    if len(train_indices) > 0:
        if train_indices.max() >= T or train_indices.min() < 0:
            raise ValueError(
                f"train_indices out of bounds: min={train_indices.min()}, "
                f"max={train_indices.max()}, expected [0, {T})"
            )
    if len(query_indices) > 0:
        if query_indices.max() >= T or query_indices.min() < 0:
            raise ValueError(
                f"query_indices out of bounds: min={query_indices.min()}, "
                f"max={query_indices.max()}, expected [0, {T})"
            )

    targets = _ensure_2d(targets)

    X_lib = manifold[train_indices]
    Y_lib = targets[train_indices]
    X_query = manifold[query_indices]

    predictions = predict(X_lib, Y_lib, X_query)
    predictions = _ensure_2d(predictions)

    score, per_target = metric(predictions, targets[query_indices])
    return score, per_target, predictions
