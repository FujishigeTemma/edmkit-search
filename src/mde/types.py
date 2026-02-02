"""Type definitions for multi-target MDE.

This module contains dataclasses and type aliases used throughout the package.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = npt.NDArray[np.floating]
RNG: TypeAlias = np.random.Generator


@dataclass
class MDEResult:
    """Result container for MDE algorithm.

    Parameters
    ----------
    selected_indices : list[int]
        Indices of selected variables (into candidates array).
    manifold : np.ndarray
        The constructed manifold of shape (T, D) where D is the number of
        selected dimensions.
    val_rhos : list[float]
        Mean validation rho at each dimension step.
    test_rhos : list[float]
        Mean test rho at each dimension step.
    val_rhos_per_target : list[np.ndarray]
        Per-target validation rhos at each dimension step.
    test_rhos_per_target : list[np.ndarray]
        Per-target test rhos at each dimension step.
    ccm_scores : list[float] | None
        CCM convergence scores if ccm_validation was enabled.
    """

    selected_indices: list[int]
    manifold: np.ndarray
    val_rhos: list[float]
    test_rhos: list[float]
    val_rhos_per_target: list[np.ndarray]
    test_rhos_per_target: list[np.ndarray]
    ccm_scores: list[float] | None = None
    val_predictions: list[np.ndarray] | None = None
    test_predictions: list[np.ndarray] | None = None


@dataclass
class CCMConvergenceResult:
    """Result container for AICc-based CCM convergence test.

    Parameters
    ----------
    converged : bool
        Whether the saturation model is favored over linear (delta_aicc >= threshold).
    score_mean : np.ndarray
        Shape (n_lib_sizes,) - mean score at each library size.
    score_var : np.ndarray
        Shape (n_lib_sizes,) - variance of scores at each library size.
    aicc_saturation : float
        AICc of the saturation model.
    aicc_linear : float
        AICc of the linear model.
    delta_aicc : float
        AICc_linear - AICc_saturation. Positive means saturation is favored.
    saturation_params : tuple[float, float, float]
        Fitted (a, b, c) for s(L) = a - b * exp(-c * L).
    linear_params : tuple[float, float]
        Fitted (alpha, beta) for s(L) = alpha + beta * L.
    """

    converged: bool
    score_mean: np.ndarray
    score_var: np.ndarray
    aicc_saturation: float
    aicc_linear: float
    delta_aicc: float
    saturation_params: tuple[float, float, float]
    linear_params: tuple[float, float]


@dataclass
class DataSplit:
    """Three-way split of time series data.

    Parameters
    ----------
    train_end : int
        Index where training data ends.
    val_end : int
        Index where validation data ends.
    train_indices : np.ndarray
        Array of indices for training set.
    val_indices : np.ndarray
        Array of indices for validation set.
    test_indices : np.ndarray
        Array of indices for test set.
    """

    train_end: int
    val_end: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
