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
    # Vectorized Pearson correlation across all targets at once
    has_nan = np.isnan(predictions).any()
    if has_nan:
        nan_mask = np.isnan(predictions)
        pred = np.where(nan_mask, 0.0, predictions)
        obs = np.where(nan_mask, 0.0, observations)
        n = (~nan_mask).sum(axis=0).astype(float)
        n = np.maximum(n, 1.0)
        pred_mean = pred.sum(axis=0) / n
        obs_mean = obs.sum(axis=0) / n
        pred_c = np.where(nan_mask, 0.0, pred - pred_mean)
        obs_c = np.where(nan_mask, 0.0, obs - obs_mean)
    else:
        pred_c = predictions - predictions.mean(axis=0)
        obs_c = observations - observations.mean(axis=0)

    cov = (pred_c * obs_c).sum(axis=0)
    pred_std = np.sqrt((pred_c**2).sum(axis=0))
    obs_std = np.sqrt((obs_c**2).sum(axis=0))
    denom = pred_std * obs_std
    per_target = np.where(denom > 0, cov / denom, 0.0)

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
    diff = predictions - observations
    sq = diff**2
    per_target = np.sqrt(np.nanmean(sq, axis=0))
    # If all NaN in a column, nanmean returns NaN -> replace with inf
    per_target = np.where(np.isnan(per_target), np.inf, per_target)

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
    diff = np.abs(predictions - observations)
    per_target = np.nanmean(diff, axis=0)
    # If all NaN in a column, nanmean returns NaN -> replace with inf
    per_target = np.where(np.isnan(per_target), np.inf, per_target)

    aggregate = float(np.mean(per_target))
    return aggregate, per_target
