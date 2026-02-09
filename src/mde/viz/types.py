"""Visualization-specific type definitions."""

from typing import NamedTuple

import numpy as np


class MDEPlotData(NamedTuple):
    """Pre-computed data for MDE result plots.

    Parameters
    ----------
    target_label : str
        Display label for the target variable(s).
    selected_var_names : list[str]
        Names of selected variables at each dimension step.
    val_scores : list[float]
        Mean validation score at each dimension step.
    test_scores : list[float]
        Mean test score at each dimension step.
    val_scores_per_target : list[np.ndarray]
        Per-target validation scores at each dimension step.
    test_scores_per_target : list[np.ndarray]
        Per-target test scores at each dimension step.
    query_indices : np.ndarray
        Indices used for the plotted split (val or test).
    observations : np.ndarray
        Ground truth values at query_indices, shape (N,) or (N, M).
    predictions_per_dim : list[np.ndarray]
        Predictions at each dimension step, each shape (N,) or (N, M).
    """

    target_label: str
    selected_var_names: list[str]
    val_scores: list[float]
    test_scores: list[float]
    val_scores_per_target: list[np.ndarray]
    test_scores_per_target: list[np.ndarray]
    query_indices: np.ndarray
    observations: np.ndarray
    predictions_per_dim: list[np.ndarray]
