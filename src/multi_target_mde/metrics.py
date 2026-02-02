"""Evaluation metrics for MDE optimization.

This module provides metric functions for evaluating prediction quality in MDE.
Each metric function takes predictions and observations and returns an aggregate
score and per-target scores.

Convention: All metrics return raw values. Use `higher_is_better` parameter
in mde() to specify comparison direction.
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

# Type alias for metric functions
# Input: predictions (N, M), observations (N, M)
# Output: (aggregate_score, per_target_scores of shape (M,))
MetricFn: TypeAlias = Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]


def mean_rho(
    predictions: np.ndarray, observations: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute mean Pearson correlation coefficient.

    Higher values indicate better predictions.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    aggregate : float
        Mean correlation across all targets.
    per_target : np.ndarray of shape (M,)
        Correlation for each target.
    """
    M = predictions.shape[1]
    per_target = np.array(
        [
            np.corrcoef(predictions[:, m], observations[:, m])[0, 1]
            if not np.isnan(predictions[:, m]).all()
            else 0.0
            for m in range(M)
        ]
    )
    per_target = np.nan_to_num(per_target, nan=0.0)
    aggregate = float(np.mean(per_target))
    return aggregate, per_target


def rmse(predictions: np.ndarray, observations: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute Root Mean Squared Error.

    Lower values indicate better predictions.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    aggregate : float
        Mean RMSE across all targets.
    per_target : np.ndarray of shape (M,)
        RMSE for each target.
    """
    M = predictions.shape[1]
    per_target = np.zeros(M)

    for m in range(M):
        valid_mask = ~np.isnan(predictions[:, m])
        if valid_mask.sum() > 0:
            mse = np.mean(
                (predictions[valid_mask, m] - observations[valid_mask, m]) ** 2
            )
            per_target[m] = np.sqrt(mse)
        else:
            per_target[m] = np.inf

    aggregate = float(np.mean(per_target))
    return aggregate, per_target


def mae(predictions: np.ndarray, observations: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute Mean Absolute Error.

    Lower values indicate better predictions.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted values where N is number of samples and M is number of targets.
    observations : np.ndarray of shape (N, M)
        Observed (ground truth) values.

    Returns
    -------
    aggregate : float
        Mean MAE across all targets.
    per_target : np.ndarray of shape (M,)
        MAE for each target.
    """
    M = predictions.shape[1]
    per_target = np.zeros(M)

    for m in range(M):
        valid_mask = ~np.isnan(predictions[:, m])
        if valid_mask.sum() > 0:
            per_target[m] = np.mean(
                np.abs(predictions[valid_mask, m] - observations[valid_mask, m])
            )
        else:
            per_target[m] = np.inf

    aggregate = float(np.mean(per_target))
    return aggregate, per_target
