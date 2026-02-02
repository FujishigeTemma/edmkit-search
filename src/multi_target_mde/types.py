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
